#!/usr/bin/env python3
"""
Camera Diagnostics Tool for ChessDrum
Tests camera detection and helps diagnose issues.
"""
import cv2
import sys
import json
import numpy as np

def test_camera_devices():
    """Test which camera devices are available."""
    print("🔍 Testing camera devices...")
    print("-" * 50)
    
    available = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                available.append((i, w, h))
                print(f"✓ Device {i}: {w}x{h}")
            else:
                print(f"✗ Device {i}: Opens but can't read frame")
            cap.release()
        else:
            print(f"✗ Device {i}: Not available")
    
    print("-" * 50)
    if available:
        print(f"\n✅ Found {len(available)} working camera(s)")
        print("\nRecommended config.json setting:")
        print(f'  "device_id": {available[0][0]}  // {available[0][1]}x{available[0][2]}')
    else:
        print("\n❌ No working cameras found")
        print("\nTroubleshooting:")
        print("  1. Check camera is connected")
        print("  2. Check camera permissions (System Preferences → Security)")
        print("  3. Try closing other apps using the camera")
    
    return available


def show_camera_feed(device_id=0):
    """Show live camera feed with diagnostics."""
    print(f"\n📹 Opening camera {device_id}...")
    cap = cv2.VideoCapture(device_id)
    
    if not cap.isOpened():
        print(f"❌ Could not open camera {device_id}")
        return
    
    print("✓ Camera opened")
    print("\nControls:")
    print("  ESC/X - Quit")
    print("  Space - Take snapshot (saved to camera_snapshot.jpg)")
    print("  Q/A - Brightness +/-")
    print("  W/S - Contrast +/-")
    print("  C - Clear manual corners")
    print("  P - Print config snippet")
    print("  Left click - Add corner (TL, TR, BR, BL)")
    print("  Right click - Undo last corner")
    print("-" * 50)
    
    brightness = 0
    contrast = 1.0
    manual_points = []
    manual_corners = None

    window_name = "ChessDrum Camera Test"

    def on_mouse(event, x, y, flags, param):
        nonlocal manual_points, manual_corners
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(manual_points) < 4:
                manual_points.append((x, y))
                if len(manual_points) == 4:
                    manual_corners = manual_points.copy()
        elif event == cv2.EVENT_RBUTTONDOWN:
            if manual_points:
                manual_points.pop()
            manual_corners = manual_points.copy() if len(manual_points) == 4 else None

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Apply adjustments
        adjusted = frame.astype('float32')
        adjusted = adjusted * contrast + brightness
        adjusted = adjusted.clip(0, 255).astype('uint8')
        
        # Show stats
        gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
        mean = gray.mean()
        std = gray.std()
        
        cv2.putText(adjusted, f"Brightness: {int(mean)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(adjusted, f"Contrast: {std:.1f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(adjusted, f"Adj: Q/A {brightness}  W/S {contrast:.1f}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(adjusted, "Click 4 corners (TL,TR,BR,BL)", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(adjusted, "SPACE snapshot | ESC/X quit | P print config", (10, adjusted.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if manual_points:
            for pt in manual_points:
                cv2.circle(adjusted, pt, 6, (0, 255, 255), -1)
            if len(manual_points) >= 2:
                pts = np.array(manual_points, dtype=np.int32)
                cv2.polylines(adjusted, [pts], len(manual_points) == 4, (0, 255, 255), 2)
        
        cv2.imshow(window_name, adjusted)
        
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('x'), ord('X')):
            break
        elif key == ord(' '):
            cv2.imwrite('camera_snapshot.jpg', adjusted)
            print("📸 Snapshot saved to camera_snapshot.jpg")
        elif key == ord('q'):
            brightness = min(100, brightness + 5)
        elif key == ord('a'):
            brightness = max(-100, brightness - 5)
        elif key == ord('w'):
            contrast = min(3.0, contrast + 0.1)
        elif key == ord('s'):
            contrast = max(0.1, contrast - 0.1)
        elif key == ord('c'):
            manual_points = []
            manual_corners = None
        elif key == ord('p'):
            if manual_corners:
                h, w = adjusted.shape[:2]
                norm = [[round(x / w, 4), round(y / h, 4)] for x, y in manual_corners]
                snippet = {
                    "camera": {
                        "brightness": brightness,
                        "contrast": round(contrast, 2),
                        "manual_corners": norm
                    }
                }
                print(json.dumps(snippet, indent=2))
            else:
                print("Manual corners not set (need 4 points)")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Camera test ended")


if __name__ == "__main__":
    print("=" * 50)
    print("  ChessDrum Camera Diagnostics 🎥")
    print("=" * 50)
    print()
    
    # Test all devices
    devices = test_camera_devices()
    
    if not devices:
        sys.exit(1)
    
    # Ask which to test
    print(f"\n🎬 Show live feed? (y/n)")
    response = input().strip().lower()
    
    if response == 'y':
        if len(devices) > 1:
            print(f"\nWhich device? ({', '.join(str(d[0]) for d in devices)})")
            try:
                device_id = int(input().strip())
            except:
                device_id = devices[0][0]
        else:
            device_id = devices[0][0]
        
        show_camera_feed(device_id)
    else:
        print("\n✅ Diagnostics complete")
