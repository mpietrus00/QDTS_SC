#!/usr/bin/env python3
"""
SIFT Flow Chimerization Tool
Based on the technique used by Florian Hecker
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata


class SIFTFlowChimerizer:
    def __init__(self, img1_path: str, img2_path: str, params: dict = None):
        self.img1 = self._load_image(img1_path)
        self.img2 = self._load_image(img2_path)
        
        # Resize img2 to match img1
        h1, w1 = self.img1.shape[:2]
        h2, w2 = self.img2.shape[:2]
        if (h1, w1) != (h2, w2):
            print(f"Image 1 size: {w1}x{h1}")
            print(f"Image 2 size: {w2}x{h2} -> resizing to {w1}x{h1}")
            self.img2 = cv2.resize(self.img2, (w1, h1), interpolation=cv2.INTER_AREA)
        
        self.height, self.width = self.img1.shape[:2]
        self.flow = None
        self.sift = cv2.SIFT_create(nfeatures=5000)  # More features
        
        self.params = {
            'warp_strength': 1.0,
            'flow_levels': 5,
            'flow_winsize': 15,
            'flow_iterations': 3,
            'sift_ratio': 0.7,
            'smoothness': 1.5,
            'passes': 1,           # Multiple warp passes for melted effect
            'turbulence': 0.0,     # Add chaos to flow
            'melt': False,         # Extreme melting mode
        }
        if params:
            self.params.update(params)

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        return img

    def compute_sift_flow(self) -> np.ndarray:
        print("Computing SIFT features...")
        gray1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        kp1, desc1 = self.sift.detectAndCompute(gray1, None)
        kp2, desc2 = self.sift.detectAndCompute(gray2, None)
        print(f"Found {len(kp1)} keypoints in image 1, {len(kp2)} in image 2")

        if desc1 is None or desc2 is None or len(kp1) < 4 or len(kp2) < 4:
            print("Using optical flow only")
            self.flow = self._compute_optical_flow(gray1, gray2)
            return self.flow

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)

        good_matches = []
        ratio = self.params['sift_ratio']
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < ratio * n.distance:
                    good_matches.append(m)
        print(f"Found {len(good_matches)} good matches")

        if len(good_matches) < 4:
            print("Using optical flow only")
            self.flow = self._compute_optical_flow(gray1, gray2)
            return self.flow

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        displacements = pts2 - pts1

        print("Interpolating dense flow field...")
        self.flow = self._interpolate_flow(pts1, displacements)
        print("Refining with optical flow...")
        self.flow = self._refine_with_optical_flow(gray1, gray2, self.flow)
        
        # Add turbulence if requested
        if self.params['turbulence'] > 0:
            self.flow = self._add_turbulence(self.flow)
        
        return self.flow

    def _compute_optical_flow(self, gray1, gray2):
        p = self.params
        return cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, p['flow_levels'], p['flow_winsize'],
            p['flow_iterations'], 7, p['smoothness'], cv2.OPTFLOW_FARNEBACK_GAUSSIAN
        )

    def _interpolate_flow(self, points, values):
        grid_x, grid_y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        flow = np.zeros((self.height, self.width, 2), dtype=np.float32)
        for i in range(2):
            interpolated = griddata(points, values[:, i], grid_points, method='linear', fill_value=0)
            flow[:, :, i] = interpolated.reshape(self.height, self.width)
        for i in range(2):
            mask = np.isnan(flow[:, :, i])
            if mask.any():
                flow[:, :, i] = np.where(mask, ndimage.generic_filter(flow[:, :, i], np.nanmean, size=5), flow[:, :, i])
        return flow

    def _refine_with_optical_flow(self, gray1, gray2, initial_flow):
        p = self.params
        return cv2.calcOpticalFlowFarneback(
            gray1, gray2, initial_flow.copy(), 0.5, max(1, p['flow_levels'] - 2),
            p['flow_winsize'], p['flow_iterations'], 5, p['smoothness'] * 0.8,
            cv2.OPTFLOW_USE_INITIAL_FLOW
        )

    def _add_turbulence(self, flow):
        """Add turbulent noise to flow field for more chaotic warping."""
        turb = self.params['turbulence']
        
        # Create multi-scale noise
        noise_x = np.zeros((self.height, self.width), dtype=np.float32)
        noise_y = np.zeros((self.height, self.width), dtype=np.float32)
        
        for scale in [1, 2, 4, 8]:
            h, w = self.height // scale, self.width // scale
            nx = np.random.randn(h, w).astype(np.float32) * (turb * 50 / scale)
            ny = np.random.randn(h, w).astype(np.float32) * (turb * 50 / scale)
            noise_x += cv2.resize(nx, (self.width, self.height))
            noise_y += cv2.resize(ny, (self.width, self.height))
        
        # Smooth the noise
        noise_x = cv2.GaussianBlur(noise_x, (0, 0), 10)
        noise_y = cv2.GaussianBlur(noise_y, (0, 0), 10)
        
        flow[:, :, 0] += noise_x
        flow[:, :, 1] += noise_y
        return flow

    def warp_image(self, img, flow, strength):
        h, w = img.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (x + flow[:, :, 0] * strength).astype(np.float32)
        map_y = (y + flow[:, :, 1] * strength).astype(np.float32)
        return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def chimerize(self, t=1.0):
        """
        Apply chimerization: warp image1 using flow derived from image2.
        t: interpolation factor (0=original, 1=fully warped)
        """
        if self.flow is None:
            self.compute_sift_flow()
        
        strength = self.params['warp_strength'] * t
        passes = self.params['passes']
        melt = self.params['melt']
        
        result = self.img1.copy()
        
        # Multiple passes for more extreme melting
        for p in range(passes):
            pass_strength = strength / passes
            if melt:
                # In melt mode, accumulate warping progressively
                pass_strength = strength * ((p + 1) / passes)
            result = self.warp_image(result, self.flow, pass_strength)
            
            if melt and p < passes - 1:
                # Slight blur between passes for melted look
                result = cv2.GaussianBlur(result, (3, 3), 0)
        
        return result

    def generate_sequence(self, num_frames=30):
        if self.flow is None:
            self.compute_sift_flow()
        
        print(f"Generating {num_frames} chimerization frames...")
        frames = []

        for i in range(num_frames):
            t = i / (num_frames - 1)
            frame = self.chimerize(t)
            frames.append(frame)
            
            if (i + 1) % 10 == 0:
                print(f"  Generated frame {i + 1}/{num_frames}")
        return frames

    def save_single(self, output_path, t=1.0):
        """Save a single chimerized image."""
        result = self.chimerize(t)
        cv2.imwrite(output_path, result)
        print(f"Saved chimerized image to: {output_path}")
        return result

    def save_gif(self, frames, output_path, fps=15):
        from PIL import Image
        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
        pil_frames[0].save(output_path, save_all=True, append_images=pil_frames[1:], duration=int(1000/fps), loop=0)
        print(f"Saved animation to: {output_path}")

    def save_video(self, frames, output_path, fps=15):
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for frame in frames:
            out.write(frame)
        out.release()
        print(f"Saved video to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SIFT Flow Chimerization Tool (Florian Hecker style)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic chimerization
  python3 sift_flow_morph.py img1.jpg img2.jpg --strength 3.0

  # Extreme melt effect (multiple passes)
  python3 sift_flow_morph.py img1.jpg img2.jpg --melt --passes 5 --strength 2.0

  # Add turbulence for chaos
  python3 sift_flow_morph.py img1.jpg img2.jpg --turbulence 0.5 --strength 3.0

  # Save single still image
  python3 sift_flow_morph.py img1.jpg img2.jpg --still --strength 4.0 -o chimerized.jpg
        """
    )
    parser.add_argument("image1", help="Image to warp")
    parser.add_argument("image2", help="Image providing spatial structure")
    parser.add_argument("-o", "--output", default="chimerized.gif", help="Output file")
    parser.add_argument("-n", "--frames", type=int, default=30, help="Number of frames")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second")
    parser.add_argument("--video", action="store_true", help="Output as MP4")
    parser.add_argument("--still", action="store_true", help="Output single image instead of animation")
    
    # Chimerization parameters
    parser.add_argument("--strength", type=float, default=1.0, help="Warp strength (try 2.0-5.0 for Hecker-style)")
    parser.add_argument("--passes", type=int, default=1, help="Number of warp passes (more=more melted)")
    parser.add_argument("--melt", action="store_true", help="Enable melt mode for extreme distortion")
    parser.add_argument("--turbulence", type=float, default=0.0, help="Add chaos (0.0-1.0)")
    parser.add_argument("--smoothness", type=float, default=1.5, help="Flow smoothness")
    parser.add_argument("--levels", type=int, default=5, help="Pyramid levels")
    parser.add_argument("--winsize", type=int, default=15, help="Window size")
    
    args = parser.parse_args()

    if not Path(args.image1).exists():
        print(f"Error: Image not found: {args.image1}")
        sys.exit(1)
    if not Path(args.image2).exists():
        print(f"Error: Image not found: {args.image2}")
        sys.exit(1)

    params = {
        'warp_strength': args.strength,
        'flow_levels': args.levels,
        'flow_winsize': args.winsize,
        'smoothness': args.smoothness,
        'passes': args.passes,
        'turbulence': args.turbulence,
        'melt': args.melt,
    }

    print(f"SIFT Flow Chimerization\n=======================")
    print(f"Source: {args.image1}")
    print(f"Structure from: {args.image2}")
    print(f"Strength: {args.strength} | Passes: {args.passes} | Melt: {args.melt} | Turbulence: {args.turbulence}\n")

    chimerizer = SIFTFlowChimerizer(args.image1, args.image2, params)
    chimerizer.compute_sift_flow()
    
    if args.still:
        # Single image output
        output = args.output if args.output.endswith(('.jpg', '.png')) else 'chimerized.jpg'
        chimerizer.save_single(output)
    else:
        # Animation output
        frames = chimerizer.generate_sequence(args.frames)
        if args.video:
            chimerizer.save_video(frames, args.output.replace('.gif', '.mp4'), args.fps)
        else:
            chimerizer.save_gif(frames, args.output, args.fps)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
