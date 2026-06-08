import argparse
from pathlib import Path

import cv2
import numpy as np


# Pragurile sunt raportate la dimensiunea imaginii, ca algoritmul sa se
# adapteze cat de cat la poze mici si mari.
MIN_STRIPE_AREA_RATIO = 0.00045

# O banda de trecere trebuie sa fie mai lunga decat lata, dar nu acceptam
# forme exagerat de subtiri care sunt de obicei linii, reflexii sau zgomot.
MIN_ASPECT_RATIO = 1.6
MAX_ASPECT_RATIO = 20.0

# O trecere de pietoni este considerata valida doar daca gasim mai multe
# marcaje compatibile, nu o singura zona alba izolata.
MIN_STRIPES_FOR_CROSSWALK = 3

# Benzile din aceeasi trecere trebuie sa aiba orientari apropiate.
ANGLE_TOLERANCE_DEGREES = 18.0

# Structura folosita pentru o banda candidata:
# (x, y, w, h, center_x, center_y, angle, long_side, short_side, area)
Stripe = tuple[int, int, int, int, float, float, float, float, float, float]


def angle_distance(angle_a: float, angle_b: float) -> float:
    """Calculeaza cea mai mica diferenta dintre doua unghiuri de linii."""
    diff = abs(angle_a - angle_b) % 180.0
    return min(diff, 180.0 - diff)


def load_image(image_path: str) -> np.ndarray:
    """Citeste imaginea de la calea primita si opreste programul daca lipseste."""
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Nu am putut citi imaginea: {image_path}")

    return image


def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construieste mastile folosite pentru izolarea marcajelor albe."""
    # Grayscale ajuta la praguri de luminozitate, HSV separa mai bine
    # saturatia, iar LAB ofera un canal de luminozitate util pentru threshold
    # adaptiv in imagini cu umbre.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    blurred_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred_l = cv2.GaussianBlur(lab[:, :, 0], (5, 5), 0)

    # Cauta zone albe: luminoase si cu saturatie mica. Combinam un prag global
    # cu unul local pentru a pastra marcaje vizibile si in zone mai intunecate.
    white_by_color = cv2.inRange(hsv, np.array([0, 0, 105]), np.array([180, 95, 255]))
    _, bright_global = cv2.threshold(blurred_gray, 175, 255, cv2.THRESH_BINARY)
    bright_local = cv2.adaptiveThreshold(
        blurred_l,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -4,
    )

    threshold = cv2.bitwise_and(white_by_color, cv2.bitwise_or(bright_local, bright_global))

    # Ignora partea de sus a imaginii, unde apar cerul, cladiri, masini sau alte
    # obiecte care pot contine suprafete albe dar nu sunt carosabil.
    height = image.shape[0]
    road_roi = np.zeros_like(threshold)
    road_roi[int(height * 0.22) :, :] = 255
    threshold = cv2.bitwise_and(threshold, road_roi)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Inchiderea morfologica uneste fragmente apropiate ale aceluiasi marcaj.
    closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # Deschiderea morfologica elimina puncte albe mici ramase dupa thresholding.
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel, iterations=1)

    return gray, threshold, cleaned


def find_candidate_stripes(mask: np.ndarray) -> list[Stripe]:
    """Gaseste contururi albe care au forma potrivita pentru benzi de trecere."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    image_area = mask.shape[0] * mask.shape[1]

    # Aria minima elimina zgomotul; dimensiunea minima elimina formele alungite,
    # dar prea mici ca sa fie credibile ca marcaje rutiere.
    min_stripe_area = max(45, image_area * MIN_STRIPE_AREA_RATIO)
    min_dimension = max(8, int(min(mask.shape[:2]) * 0.025))

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_stripe_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if w == 0 or h == 0:
            continue

        # minAreaRect estimeaza dreptunghiul rotit al conturului. Din el luam
        # axa lunga si unghiul, indiferent daca banda este orizontala sau oblica.
        rect = cv2.minAreaRect(contour)
        (center_x, center_y), (stripe_width, stripe_height), angle = rect
        short_side = min(stripe_width, stripe_height)
        long_side = max(stripe_width, stripe_height)

        if short_side <= 0:
            continue

        aspect_ratio = long_side / short_side
        long_axis_angle = angle + 90.0 if stripe_width < stripe_height else angle
        long_axis_angle %= 180.0

        # Accepta benzi orizontale, verticale sau oblice, atata timp cat sunt
        # suficient de alungite si destul de mari.
        if (
            MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO
            and long_side >= min_dimension
            and short_side >= 5
        ):
            candidates.append(
                (
                    x,
                    y,
                    w,
                    h,
                    float(center_x),
                    float(center_y),
                    float(long_axis_angle),
                    float(long_side),
                    float(short_side),
                    float(area),
                )
            )

    return candidates


def build_stripe_from_contour(contour: np.ndarray) -> Stripe | None:
    """Transforma un contur OpenCV in structura Stripe folosita de algoritm."""
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)

    if area <= 0 or w <= 0 or h <= 0:
        return None

    rect = cv2.minAreaRect(contour)
    (center_x, center_y), (stripe_width, stripe_height), angle = rect
    short_side = min(stripe_width, stripe_height)
    long_side = max(stripe_width, stripe_height)

    if short_side <= 0:
        return None

    long_axis_angle = angle + 90.0 if stripe_width < stripe_height else angle
    long_axis_angle %= 180.0

    return (
        x,
        y,
        w,
        h,
        float(center_x),
        float(center_y),
        float(long_axis_angle),
        float(long_side),
        float(short_side),
        float(area),
    )


def is_dense_non_road_stripe_texture(
    image: np.ndarray,
    mask: np.ndarray,
    crosswalk_box: tuple[int, int, int, int] | None,
    stripes: list[Stripe],
) -> bool:
    """Detecteaza texturi cu dungi care seamana cu o trecere, dar nu sunt drum."""
    # Filtrul este important pentru imagini precum haine, materiale textile sau
    # modele alb-negru. Acestea pot avea multe dungi paralele, dar contextul din
    # jur nu arata ca asfaltul.
    if crosswalk_box is None or not stripes:
        return False

    image_height, image_width = image.shape[:2]
    image_area = image_height * image_width
    x, y, w, h = crosswalk_box

    if w <= 0 or h <= 0:
        return True

    padding_x = int(w * 0.18)
    padding_y = int(h * 0.22)
    x_min = max(0, x - padding_x)
    y_min = max(0, y - padding_y)
    x_max = min(image_width, x + w + padding_x)
    y_max = min(image_height, y + h + padding_y)

    if x_max <= x_min or y_max <= y_min:
        return True

    roi = image[y_min:y_max, x_min:x_max]
    roi_mask = mask[y_min:y_max, x_min:x_max]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    context_pixels = roi_mask == 0
    context_count = int(np.count_nonzero(context_pixels))

    if context_count == 0:
        return True

    # Contextul "ca de drum" inseamna pixeli fara marcaj, cu saturatie mica si
    # luminozitate medie. Daca aproape tot fundalul este negru/alb extrem,
    # probabil privim o textura, nu carosabil.
    road_like_context = (
        context_pixels
        & (saturation < 90)
        & (gray >= 35)
        & (gray <= 215)
    )
    road_like_ratio = float(np.count_nonzero(road_like_context)) / context_count

    extreme_ratio = float(np.count_nonzero((gray < 28) | (gray > 235))) / gray.size
    white_ratio = float(np.count_nonzero(roi_mask > 0)) / roi_mask.size

    short_sides = np.array([stripe[8] for stripe in stripes], dtype=np.float32)
    median_short_side = float(np.median(short_sides))
    thin_stripe_limit = max(5.0, min(image_height, image_width) * 0.018)

    selected_area = sum(stripe[9] for stripe in stripes)
    selected_area_ratio = selected_area / float(image_area)
    box_area = w * h
    box_area_ratio = box_area / float(image_area)
    selected_fill_in_box = selected_area / float(box_area)
    many_thin_stripes = len(stripes) >= 8 and median_short_side <= thin_stripe_limit
    stripe_angles = np.array([stripe[6] for stripe in stripes], dtype=np.float32)
    vertical_stripes = sum(
        min(angle_distance(float(angle), 90.0), angle_distance(float(angle), 0.0)) < 18.0
        for angle in stripe_angles
    )
    mostly_axis_aligned = vertical_stripes >= max(3, int(len(stripes) * 0.7))

    # Respinge box-urile mari formate din putine dungi rare, deoarece de obicei
    # apar din obiecte apropiate de camera sau din suprafete texturate.
    sparse_oversized_detection = (
        box_area_ratio > 0.30
        and len(stripes) <= 4
        and selected_fill_in_box < 0.16
    )

    # Hainele cu dungi pot avea marcaje subtiri, paralele si contrast mare.
    clothing_like_detection = (
        box_area_ratio > 0.12
        and len(stripes) <= 6
        and median_short_side <= max(14.0, min(image_height, image_width) * 0.045)
        and mostly_axis_aligned
        and road_like_ratio < 0.18
        and (extreme_ratio > 0.30 or selected_fill_in_box < 0.20)
    )

    oversized_pattern = (
        box_area_ratio > 0.42
        and len(stripes) >= 6
        and selected_area_ratio > 0.035
    )

    weak_road_context = road_like_ratio < 0.24 and extreme_ratio > 0.42
    high_contrast_texture = extreme_ratio > 0.62 and white_ratio > 0.12

    return (
        sparse_oversized_detection or
        clothing_like_detection or
        (many_thin_stripes and weak_road_context)
        or (many_thin_stripes and high_contrast_texture)
        or (oversized_pattern and weak_road_context)
    )


def detect_by_white_marking_density(
    mask: np.ndarray,
    image_shape: tuple[int, ...],
) -> tuple[bool, tuple[int, int, int, int] | None, list[Stripe]]:
    """Fallback pentru marcaje unite, sterse sau slab luminate."""
    # Uneori benzile nu apar ca dreptunghiuri separate dupa thresholding. In
    # acest caz incercam o detectie mai relaxata, bazata pe densitatea zonelor
    # albe din partea inferioara a imaginii.
    image_height, image_width = image_shape[:2]
    image_area = image_height * image_width

    relaxed = mask.copy()
    relaxed[: int(image_height * 0.18), :] = 0

    # Kernelul de inchidere este dimensionat dupa imagine, ca sa uneasca
    # fragmente apropiate fara sa transforme toata scena intr-o singura pata.
    join_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(5, image_width // 45), max(3, image_height // 80)),
    )
    relaxed = cv2.morphologyEx(relaxed, cv2.MORPH_CLOSE, join_kernel, iterations=1)

    contours, _ = cv2.findContours(relaxed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = max(35, image_area * 0.00022)
    components: list[Stripe] = []

    for contour in contours:
        stripe = build_stripe_from_contour(contour)

        if stripe is None:
            continue

        x, y, w, h, _, center_y, _, long_side, short_side, area = stripe
        fill_ratio = area / float(w * h)

        if center_y < image_height * 0.22:
            continue

        if area < min_area:
            continue

        if long_side < max(8, min(image_shape[:2]) * 0.018):
            continue

        if short_side < 3 and fill_ratio < 0.2:
            continue

        components.append(stripe)

    if not components:
        return False, None, []

    # Luam cele mai mari componente fiindca ele sunt cel mai probabil marcaje
    # rutiere, iar punctele mici ramase sunt zgomot.
    components.sort(key=lambda stripe: stripe[9], reverse=True)
    selected = components[: min(16, len(components))]

    if len(selected) == 1 and selected[0][9] < image_area * 0.018:
        return False, None, []

    if len(selected) == 2 and sum(stripe[9] for stripe in selected) < image_area * 0.006:
        return False, None, []

    xs = [stripe[0] for stripe in selected]
    ys = [stripe[1] for stripe in selected]
    x2s = [stripe[0] + stripe[2] for stripe in selected]
    y2s = [stripe[1] + stripe[3] for stripe in selected]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(x2s)
    y_max = max(y2s)
    width = x_max - x_min
    height = y_max - y_min

    # Zona finala trebuie sa fie suficient de lata si inalta ca sa reprezinte
    # o trecere vizibila in imagine.
    if width < image_width * 0.12 or height < image_height * 0.035:
        return False, None, []

    if width * height > image_area * 0.78:
        return False, None, []

    padding_x = int(width * 0.06)
    padding_y = int(height * 0.12)

    x_min = max(0, x_min - padding_x)
    y_min = max(0, y_min - padding_y)
    x_max = min(image_width - 1, x_max + padding_x)
    y_max = min(image_height - 1, y_max + padding_y)

    return True, (x_min, y_min, x_max - x_min, y_max - y_min), selected


def group_stripes(
    stripes: list[Stripe],
    image_shape: tuple[int, ...],
) -> tuple[bool, tuple[int, int, int, int] | None, list[Stripe]]:
    """Grupeaza benzile candidate si decide daca formeaza o trecere."""
    if len(stripes) < MIN_STRIPES_FOR_CROSSWALK:
        return False, None, []

    image_height, image_width = image_shape[:2]
    image_area = image_height * image_width
    max_cluster_height = image_height * 0.48
    max_center_distance = image_width * 0.95

    # Pastreaza marcajele mai importante. Daca filtrarea devine prea stricta,
    # revine la lista initiala pentru a nu rata trecerile mai sterse.
    areas = np.array([stripe[9] for stripe in stripes], dtype=np.float32)
    strong_area = max(image_area * MIN_STRIPE_AREA_RATIO, float(np.percentile(areas, 45)) * 0.55)
    filtered = [stripe for stripe in stripes if stripe[9] >= strong_area]

    if len(filtered) < MIN_STRIPES_FOR_CROSSWALK:
        filtered = stripes

    best_cluster: list[Stripe] = []
    best_score = -1.0

    # Fiecare banda este folosita ca "ancora". Cautam langa ea alte benzi cu
    # unghi, dimensiune si pozitie compatibile, apoi pastram cel mai bun grup.
    for seed in filtered:
        _, _, _, _, seed_cx, seed_cy, seed_angle, seed_long, _, _ = seed
        theta = np.deg2rad(seed_angle)
        long_axis = np.array([np.cos(theta), np.sin(theta)])
        seed_center = np.array([seed_cx, seed_cy])
        seed_projection = float(seed_center @ long_axis)
        # Proiectia pe axa lunga ajuta sa grupam benzi care se afla pe aceeasi
        # directie generala, inclusiv cand trecerea este vazuta in perspectiva.
        axis_tolerance = max(seed_long * 0.55, image_height * 0.09, image_width * 0.06)

        cluster = []
        for stripe in filtered:
            _, _, _, _, cx, cy, angle, long_side, short_side, _ = stripe
            center = np.array([cx, cy])
            center_distance = np.linalg.norm(center - seed_center)
            projection_distance = abs(float(center @ long_axis) - seed_projection)
            similar_size = 0.35 <= (long_side / max(seed_long, 1.0)) <= 2.4

            if (
                angle_distance(angle, seed_angle) <= ANGLE_TOLERANCE_DEGREES
                and center_distance <= max_center_distance
                and projection_distance <= axis_tolerance
                and similar_size
                and short_side >= 5
            ):
                cluster.append(stripe)

        if len(cluster) < MIN_STRIPES_FOR_CROSSWALK:
            continue

        ys = [stripe[1] for stripe in cluster]
        y2s = [stripe[1] + stripe[3] for stripe in cluster]
        xs = [stripe[0] for stripe in cluster]
        x2s = [stripe[0] + stripe[2] for stripe in cluster]
        cluster_height = max(y2s) - min(ys)
        cluster_width = max(x2s) - min(xs)

        if cluster_height > max_cluster_height:
            continue

        long_sides = np.array([stripe[7] for stripe in cluster], dtype=np.float32)
        angles = np.array([stripe[6] for stripe in cluster], dtype=np.float32)
        area_sum = sum(stripe[9] for stripe in cluster)
        coverage = cluster_width * cluster_height
        median_long_side = float(np.median(long_sides))

        if coverage < image_area * 0.01:
            continue

        angle_penalty = sum(angle_distance(float(angle), float(np.median(angles))) for angle in angles)
        size_penalty = float(np.std(long_sides))
        # Scorul favorizeaza grupuri cu multe benzi, arie mare si unghiuri
        # apropiate, dar penalizeaza grupurile prea inalte sau incoerente.
        score = (
            len(cluster) * 80000
            + area_sum * 3.0
            + coverage * 0.4
            + median_long_side * 500
            - cluster_height * 350
            - angle_penalty * 2500
            - size_penalty * 450
        )

        if score > best_score:
            best_score = score
            best_cluster = cluster

    if len(best_cluster) < MIN_STRIPES_FOR_CROSSWALK:
        return False, None, []

    xs = [stripe[0] for stripe in best_cluster]
    ys = [stripe[1] for stripe in best_cluster]
    x2s = [stripe[0] + stripe[2] for stripe in best_cluster]
    y2s = [stripe[1] + stripe[3] for stripe in best_cluster]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(x2s)
    y_max = max(y2s)

    width = x_max - x_min
    height = y_max - y_min

    # Verifica dimensiunea zonei detectate inainte de a adauga padding.
    if width <= 0 or height <= 0:
        return False, None, []

    padding_x = int(width * 0.04)
    padding_y = int(height * 0.08)

    x_min = max(0, x_min - padding_x)
    y_min = max(0, y_min - padding_y)
    x_max = min(image_width - 1, x_max + padding_x)
    y_max = min(image_height - 1, y_max + padding_y)

    return True, (x_min, y_min, x_max - x_min, y_max - y_min), best_cluster


def group_large_road_markings(
    stripes: list[Stripe],
    image_shape: tuple[int, ...],
) -> tuple[bool, tuple[int, int, int, int] | None, list[Stripe]]:
    """Fallback pentru poze cu perspectiva puternica sau marcaje foarte mari."""
    # In unele imagini apropiate de drum apar doar 2-3 benzi late, iar regula
    # principala cu minim 3 marcaje similare poate fi prea conservatoare.
    if len(stripes) < 2:
        return False, None, []

    image_height, image_width = image_shape[:2]
    image_area = image_height * image_width
    selected: list[Stripe] = []

    for stripe in stripes:
        x, y, w, h, _, center_y, _, long_side, short_side, area = stripe
        fill_ratio = area / float(w * h)

        # Pastram doar marcaje aflate in zona probabila a drumului si eliminam
        # formele foarte mari care acopera prea mult din imagine.
        if center_y < image_height * 0.33 or center_y > image_height * 0.88:
            continue

        if w > image_width * 0.82 and h > image_height * 0.16:
            continue

        if h > image_height * 0.42:
            continue

        if area < image_area * 0.0012:
            continue

        if long_side < image_width * 0.06 and long_side < image_height * 0.12:
            continue

        if fill_ratio < 0.22 and short_side < 10:
            continue

        selected.append(stripe)

    if len(selected) < 2:
        return False, None, []

    # Cele mai mari marcaje contribuie cel mai mult la box-ul final.
    selected.sort(key=lambda stripe: stripe[9], reverse=True)
    selected = selected[:10]

    total_area = sum(stripe[9] for stripe in selected)
    if len(selected) < 3 and total_area < image_area * 0.018:
        return False, None, []

    xs = [stripe[0] for stripe in selected]
    ys = [stripe[1] for stripe in selected]
    x2s = [stripe[0] + stripe[2] for stripe in selected]
    y2s = [stripe[1] + stripe[3] for stripe in selected]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(x2s)
    y_max = max(y2s)
    width = x_max - x_min
    height = y_max - y_min

    if width < image_width * 0.20 or height < image_height * 0.05:
        return False, None, []

    if width * height > image_area * 0.65:
        return False, None, []

    padding_x = int(width * 0.04)
    padding_y = int(height * 0.10)

    x_min = max(0, x_min - padding_x)
    y_min = max(0, y_min - padding_y)
    x_max = min(image_width - 1, x_max + padding_x)
    y_max = min(image_height - 1, y_max + padding_y)

    return True, (x_min, y_min, x_max - x_min, y_max - y_min), selected


def draw_results(
    image: np.ndarray,
    stripes: list[Stripe],
    detected: bool,
    crosswalk_box: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """Deseneaza pe imagine benzile folosite si rezultatul final."""
    result = image.copy()

    # Verde: benzile candidate care au fost folosite in decizia finala.
    for x, y, w, h, *_ in stripes:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if detected and crosswalk_box is not None:
        x, y, w, h = crosswalk_box

        # Rosu: dreptunghiul care incadreaza trecerea detectata.
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 3)

        cv2.putText(
            result,
            "Crosswalk detected",
            (x, max(y - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            result,
            "No crosswalk detected",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return result


def detect_crosswalk(image: np.ndarray) -> dict[str, np.ndarray | list | bool | tuple | None]:
    """Ruleaza tot procesul de detectie si intoarce imaginile intermediare."""
    gray, threshold, cleaned = preprocess_image(image)
    stripes = find_candidate_stripes(cleaned)

    # Incercarea principala cauta mai multe benzi coerente.
    detected, crosswalk_box, selected_stripes = group_stripes(stripes, image.shape)

    # Fallback pentru cazurile in care marcajele sunt mari sau deformate de
    # perspectiva si nu se potrivesc perfect cu gruparea principala.
    broad_detected, broad_box, broad_stripes = group_large_road_markings(stripes, image.shape)

    # Daca fallback-ul gaseste o zona mai convingatoare, il folosim in locul
    # rezultatului principal.
    if broad_detected and (
        not detected
        or len(broad_stripes) > len(selected_stripes)
        or (
            broad_box is not None
            and crosswalk_box is not None
            and broad_box[2] * broad_box[3] > crosswalk_box[2] * crosswalk_box[3] * 1.35
        )
    ):
        detected, crosswalk_box, selected_stripes = broad_detected, broad_box, broad_stripes

    # Ultima incercare foloseste densitatea zonelor albe, utila pentru marcaje
    # rupte, sterse sau unite intr-o singura componenta.
    if not detected:
        detected, crosswalk_box, selected_stripes = detect_by_white_marking_density(
            cleaned,
            image.shape,
        )

    # Dupa detectie aplicam un filtru de siguranta pentru a respinge texturi cu
    # dungi care nu se afla intr-un context de drum.
    if detected and is_dense_non_road_stripe_texture(
        image,
        cleaned,
        crosswalk_box,
        selected_stripes,
    ):
        detected = False
        crosswalk_box = None
        selected_stripes = []

    result = draw_results(image, selected_stripes, detected, crosswalk_box)

    return {
        "gray": gray,
        "threshold": threshold,
        "cleaned": cleaned,
        "stripes": selected_stripes,
        "detected": detected,
        "crosswalk_box": crosswalk_box,
        "result": result,
    }


def save_outputs(
    output_dir: Path,
    image_name: str,
    gray: np.ndarray,
    threshold: np.ndarray,
    cleaned: np.ndarray,
    result: np.ndarray,
) -> None:
    """Salveaza imaginile intermediare si imaginea finala in folderul de output."""
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_name).stem

    # Prefixele numerice pastreaza ordinea pipeline-ului in folderul rezultat.
    cv2.imwrite(str(output_dir / f"{stem}_01_gray.jpg"), gray)
    cv2.imwrite(str(output_dir / f"{stem}_02_threshold.jpg"), threshold)
    cv2.imwrite(str(output_dir / f"{stem}_03_cleaned_mask.jpg"), cleaned)
    cv2.imwrite(str(output_dir / f"{stem}_04_result.jpg"), result)


def parse_arguments() -> argparse.Namespace:
    """Citeste argumentele primite din terminal."""
    parser = argparse.ArgumentParser(
        description="Detectarea trecerilor de pietoni folosind OpenCV."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Calea catre imaginea de intrare.",
    )

    parser.add_argument(
        "--output",
        default="output",
        help="Folderul in care se salveaza rezultatele.",
    )

    return parser.parse_args()


def main() -> None:
    """Punctul de intrare al programului."""
    args = parse_arguments()

    image_path = Path(args.image)
    output_dir = Path(args.output)

    image = load_image(str(image_path))

    # detect_crosswalk intoarce atat decizia finala, cat si imaginile necesare
    # pentru a intelege vizual fiecare pas al procesarii.
    detection_result = detect_crosswalk(image)

    save_outputs(
        output_dir=output_dir,
        image_name=image_path.name,
        gray=detection_result["gray"],
        threshold=detection_result["threshold"],
        cleaned=detection_result["cleaned"],
        result=detection_result["result"],
    )

    print("Detectie finalizata.")
    print(f"Imagine analizata: {image_path}")

    if detection_result["detected"]:
        print("Rezultat: trecere de pietoni detectata.")
        print(f"Zona detectata: {detection_result['crosswalk_box']}")
        print(f"Marcaje folosite: {len(detection_result['stripes'])}")
    else:
        print("Rezultat: nu a fost detectata o trecere de pietoni.")

    print(f"Rezultatele au fost salvate in: {output_dir}")


if __name__ == "__main__":
    main()
