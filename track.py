import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    corners: np.ndarray
    method: str
    inliers: int = 0


def estimate_corners(source, target, corners, shape, min_inliers, motion='projective'):
    if len(source) < min_inliers:
        return None
    if motion == 'projective':
        matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    else:
        estimator = cv2.estimateAffinePartial2D if motion == 'similarity' else cv2.estimateAffine2D
        matrix, mask = estimator(source, target, method=cv2.RANSAC, ransacReprojThreshold=1.8)
    if matrix is None:
        return None
    inliers = mask.ravel().astype(bool)
    count = int(inliers.sum())
    if count < min_inliers or count / len(source) < 0.35:
        return None

    # A cluster on a small patch cannot reliably predict the object's corners.
    support = cv2.contourArea(cv2.convexHull(source[inliers]))
    if support < 0.045 * cv2.contourArea(corners):
        return None

    if motion == 'projective':
        projected = cv2.perspectiveTransform(corners[None], matrix)[0]
    else:
        projected = cv2.transform(corners[None], matrix)[0]
    if not np.isfinite(projected).all():
        return None
    height, width = shape
    area = cv2.contourArea(projected, oriented=True)
    edges = np.linalg.norm(np.roll(projected, -1, axis=0) - projected, axis=1)
    if (not cv2.isContourConvex(projected) or not 90 <= area <= 4 * width * height
            or edges.min() < 6 or max(edges[0] / edges[2], edges[2] / edges[0],
                                      edges[1] / edges[3], edges[3] / edges[1]) > 3.5):
        return None
    image = np.float32([[0, 0], [width - 1, 0],
                        [width - 1, height - 1], [0, height - 1]])
    visible, _ = cv2.intersectConvexConvex(projected, image)
    if visible < max(75, 0.15 * area):
        return None
    return projected, count


class PlanarTracker:
    def __init__(self, reference, initial_corners):
        width = round(max(np.linalg.norm(initial_corners[1] - initial_corners[0]),
                          np.linalg.norm(initial_corners[2] - initial_corners[3]))) + 1
        height = round(max(np.linalg.norm(initial_corners[3] - initial_corners[0]),
                           np.linalg.norm(initial_corners[2] - initial_corners[1]))) + 1
        self.corners = np.float32([[0, 0], [width - 1, 0],
                                   [width - 1, height - 1], [0, height - 1]])
        self.initial_corners = initial_corners
        self.sift = cv2.SIFT_create(nfeatures=1800, contrastThreshold=0.012)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        transform = cv2.getPerspectiveTransform(initial_corners, self.corners)
        crop = cv2.warpPerspective(reference, transform, (width, height))
        self.patch_size = (80, max(24, min(80, round(80 * height / width))))
        patch_width, patch_height = self.patch_size
        self.patch_corners = np.float32([[0, 0], [patch_width - 1, 0],
                                         [patch_width - 1, patch_height - 1], [0, patch_height - 1]])
        self.reference_patch = cv2.resize(crop, self.patch_size, interpolation=cv2.INTER_AREA)
        self.templates = []
        for scale in (1, 0.5, 0.25, 0.125):
            size = (round(width * scale), round(height * scale))
            if min(size) < 24:
                continue
            image = crop if scale == 1 else cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
            keypoints, descriptors = self.sift.detectAndCompute(image, None)
            if len(keypoints) >= 8:
                factors = np.float32([width / size[0], height / size[1]])
                points = (np.float32([point.pt for point in keypoints]) + 0.5) * factors - 0.5
                self.templates.append((points, descriptors))
        if not self.templates:
            raise ValueError('The reference region needs at least 8 SIFT keypoints.')
        self.previous = reference
        self.previous_corners = self.initial_corners

    def reference_similarity(self, gray, corners):
        matrix = cv2.getPerspectiveTransform(corners, self.patch_corners)
        patch = cv2.warpPerspective(gray, matrix, self.patch_size)
        visible = cv2.warpPerspective(np.ones_like(gray), matrix, self.patch_size,
                                      flags=cv2.INTER_NEAREST).astype(bool)
        if visible.mean() < 0.55:
            return 0.0
        template = self.reference_patch[visible].astype(np.float32)
        samples = patch[visible].astype(np.float32)
        template -= template.mean()
        samples -= samples.mean()
        norm = np.linalg.norm(template) * np.linalg.norm(samples)
        return np.dot(template, samples) / norm if norm > 0 else 0.0

    def match_reference(self, gray):
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None or len(descriptors) < 2:
            return None
        unique = {}
        for points, template in self.templates:
            for first, second in self.matcher.knnMatch(template, descriptors, k=2):
                if first.distance >= 0.75 * second.distance:
                    continue
                ratio = first.distance / second.distance
                previous = unique.get(first.trainIdx)
                if previous is None or ratio < previous[0]:
                    unique[first.trainIdx] = (ratio, points[first.queryIdx])
        source = np.float32([entry[1] for entry in unique.values()])
        target = np.float32([keypoints[index].pt for index in unique])
        return estimate_corners(source, target, self.corners, gray.shape, 8)

    def follow_points(self, gray):
        if self.previous_corners is None:
            return None
        area = cv2.contourArea(self.previous_corners)
        small = area < 2200
        minimum = 5 if small else 7
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, np.rint(self.previous_corners).astype(np.int32), 255)
        border = 5 if small else 9
        mask = cv2.erode(mask, np.ones((border, border), np.uint8))
        points = cv2.goodFeaturesToTrack(self.previous, 150, 0.012, 3, mask=mask)
        if points is None or len(points) < minimum:
            return None
        options = dict(winSize=(13, 13) if small else (21, 21), maxLevel=3)
        moved, found, errors = cv2.calcOpticalFlowPyrLK(self.previous, gray, points, None, **options)
        if moved is None:
            return None
        valid = found.ravel().astype(bool) & (errors.ravel() < 28)
        points, moved = points[valid], moved[valid]
        if len(points) < minimum:
            return None
        returned, back, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, moved, None, **options)
        if returned is None:
            return None
        valid = back.ravel().astype(bool) & (np.linalg.norm(returned - points, axis=2).ravel() < 1.25)
        result = estimate_corners(points[valid, 0], moved[valid, 0], self.previous_corners,
                                  gray.shape, minimum, motion='similarity' if small else 'affine')
        if result is not None:
            corners, _ = result
            scale = cv2.contourArea(corners) / area
            if 0.60 <= scale <= 1.65 and self.reference_similarity(gray, corners) >= 0.48:
                return result
        return None

    def update(self, gray):
        result = self.match_reference(gray)
        method = 'sift'
        if result is None or result[1] < 24:
            flow = self.follow_points(gray)
            if flow is not None and (result is None or self.reference_similarity(gray, flow[0])
                                     > self.reference_similarity(gray, result[0]) + 0.08):
                result = flow
                method = 'flow'
        self.previous = gray
        self.previous_corners = None if result is None else result[0]
        return None if result is None else Detection(result[0], method, result[1])


def annotate(frame, detection, label):
    corners = np.rint(detection.corners).astype(np.int32)
    color = (55, 235, 90)
    cv2.polylines(frame, [corners], True, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.polylines(frame, [corners], True, color, 2, cv2.LINE_AA)
    x, top = corners[np.argmin(corners[:, 1])]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), baseline = cv2.getTextSize(label, font, 0.55, 1)
    x = int(np.clip(x, 0, max(0, frame.shape[1] - width - 9)))
    y = int(np.clip(top - 5, height + 5, frame.shape[0] - baseline - 3))
    cv2.rectangle(frame, (x, y - height - 4), (x + width + 8, y + baseline + 2), (20, 20, 20), -1)
    cv2.putText(frame, label, (x + 4, y), font, 0.55, color, 1, cv2.LINE_AA)


def reference_corners(shape, roi, quad):
    height, width = shape
    if quad is not None:
        corners = np.float32(quad).reshape(4, 2)
        x, y = corners.T
        if (not np.all((0 <= x) & (x < width) & (0 <= y) & (y < height))
                or not cv2.isContourConvex(corners)
                or cv2.contourArea(corners, oriented=True) < 4):
            raise ValueError('Quad must be convex and listed clockwise inside the first frame.')
        return corners
    x, y, w, h = roi if roi is not None else (0, 0, width, height)
    if not (0 <= x < width and 0 <= y < height and w > 1 and h > 1
            and x + w <= width and y + h <= height):
        raise ValueError('ROI must be a rectangle inside the first frame.')
    return np.float32([[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]])


def track_video(input_path, output_path, label, roi, quad=None):
    if not input_path.is_file():
        raise ValueError(f'Input video does not exist: {input_path}')
    if input_path.resolve() == output_path.resolve():
        raise ValueError('Input and output must be different files.')
    csv_path = output_path.with_suffix('.csv')
    if csv_path.resolve() == input_path.resolve():
        raise ValueError('The CSV path would overwrite the input.')

    capture = cv2.VideoCapture(str(input_path))
    writer = None
    counts = Counter()
    try:
        success, frame = capture.read()
        if not success:
            raise ValueError(f'Cannot read the first frame: {input_path}')
        height, width = frame.shape[:2]
        corners = reference_corners((height, width), roi, quad)
        tracker = PlanarTracker(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), corners)
        detection = Detection(tracker.initial_corners, 'reference')

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('The input video has no valid frame rate.')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        codec = 'mp4v' if output_path.suffix.lower() == '.mp4' else 'MJPG'
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
        if not writer.isOpened():
            raise ValueError(f'Cannot open the output video: {output_path}')

        with csv_path.open('w', encoding='utf-8', newline='') as stream:
            log = csv.writer(stream)
            columns = [f'{axis}{index}' for index in range(1, 5) for axis in ('x', 'y')]
            log.writerow(['frame', 'method', 'inliers', *columns])
            index = 0
            while success:
                if detection is None:
                    counts['lost'] += 1
                    log.writerow([index, 'lost', 0, *([''] * 8)])
                else:
                    counts[detection.method] += 1
                    annotate(frame, detection, label)
                    log.writerow([index, detection.method, detection.inliers,
                                  *np.round(detection.corners.ravel(), 2)])
                writer.write(frame)
                index += 1
                success, frame = capture.read()
                if success:
                    detection = tracker.update(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return counts


def main():
    project = Path(__file__).resolve().parent
    sample = project / 'test-videos' / 'my_video.MOV'
    parser = argparse.ArgumentParser(description='Track a planar object from the first video frame.')
    parser.add_argument('input', type=Path, nargs='?', default=sample,
                        help='Input video (default: test-videos/my_video.MOV)')
    parser.add_argument('-o', '--output', type=Path,
                        help='Annotated .mp4 or .avi video (default: output/<input>-tracked.mp4)')
    parser.add_argument('--label', help='ASCII label printed above the outline')
    region = parser.add_mutually_exclusive_group()
    region.add_argument('--roi', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'),
                        help='Axis-aligned object bounds on the first frame. Defaults to the entire frame.')
    region.add_argument('--quad', type=int, nargs=8,
                        metavar=('X1', 'Y1', 'X2', 'Y2', 'X3', 'Y3', 'X4', 'Y4'),
                        help='Object corners clockwise from the top left on the first frame.')
    args = parser.parse_args()
    sample_selected = args.input.resolve() == sample
    output = args.output or project / 'output' / f'{args.input.stem}-tracked.mp4'
    label = args.label if args.label is not None else ('Book' if sample_selected else 'Object')
    quad = args.quad
    if sample_selected and args.roi is None and quad is None:
        quad = (457, 220, 742, 219, 775, 672, 442, 683)
    if output.suffix.lower() not in {'.mp4', '.avi'}:
        parser.error('output must have the .mp4 or .avi extension')
    if not label or not label.isascii():
        parser.error('label must be nonempty ASCII text')
    try:
        counts = track_video(args.input, output, label, args.roi, quad)
    except (OSError, ValueError, cv2.error) as error:
        parser.exit(1, f'{error}\n')
    print(f'{args.input.name}: {sum(counts.values())} frames, '
          f'reference={counts["reference"]}, sift={counts["sift"]}, '
          f'flow={counts["flow"]}, lost={counts["lost"]}')


if __name__ == '__main__':
    main()
