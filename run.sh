#!/bin/bash

# ChessDrum Launcher
# Interactive script to run ChessDrum with or without camera

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       🎵 ChessDrum Launcher 🎵        ║${NC}"
echo -e "${CYAN}╔════════════════════════════════════════╗${NC}"
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    echo -e "${GREEN}✓${NC} Activating virtual environment..."
    source venv/bin/activate
else
    echo -e "${RED}✗${NC} Virtual environment not found!"
    echo "  Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Function to detect available cameras
detect_cameras() {
    echo -e "${YELLOW}🔍 Detecting available cameras...${NC}"
    
    # Create a temporary Python script to detect cameras
    python3 << 'EOF'
import cv2
import sys

cameras = []
for i in range(10):  # Check first 10 camera indices
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            cameras.append(i)
        cap.release()

if cameras:
    print("CAMERAS:" + ",".join(map(str, cameras)))
else:
    print("CAMERAS:")
sys.exit(0)
EOF
    
    camera_output=$?
    return $camera_output
}

# Camera mode selection
echo ""
echo -e "${BLUE}Select mode:${NC}"
echo "  1) Virtual mode (no camera)"
echo "  2) Camera mode"
echo ""
read -p "Enter choice [1-2]: " mode_choice

if [ "$mode_choice" == "1" ]; then
    # Virtual mode
    echo ""
    echo -e "${GREEN}▶${NC} Starting ChessDrum in ${YELLOW}VIRTUAL MODE${NC}..."
    echo ""
    python3 src/main.py "$@"
    
elif [ "$mode_choice" == "2" ]; then
    # Camera mode - detect cameras
    camera_info=$(python3 << 'EOF'
import cv2
cameras = []
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            cameras.append(i)
        cap.release()
print(",".join(map(str, cameras)))
EOF
)
    
    if [ -z "$camera_info" ]; then
        echo ""
        echo -e "${RED}✗${NC} No cameras detected!"
        echo -e "  Falling back to ${YELLOW}VIRTUAL MODE${NC}..."
        echo ""
        python3 src/main.py "$@"
    else
        IFS=',' read -ra CAMERAS <<< "$camera_info"
        
        echo ""
        echo -e "${GREEN}✓${NC} Found ${#CAMERAS[@]} camera(s): ${CAMERAS[*]}"
        echo ""
        
        if [ ${#CAMERAS[@]} -eq 1 ]; then
            # Only one camera, use it automatically
            selected_cam=${CAMERAS[0]}
            echo -e "${GREEN}▶${NC} Using camera ${YELLOW}${selected_cam}${NC}"
        else
            # Multiple cameras, let user choose
            echo -e "${BLUE}Available cameras:${NC}"
            for cam in "${CAMERAS[@]}"; do
                echo "  • Camera $cam"
            done
            echo ""
            read -p "Select camera index: " selected_cam
            
            # Validate selection
            valid=0
            for cam in "${CAMERAS[@]}"; do
                if [ "$selected_cam" == "$cam" ]; then
                    valid=1
                    break
                fi
            done
            
            if [ $valid -eq 0 ]; then
                echo -e "${RED}✗${NC} Invalid camera selection. Using camera ${CAMERAS[0]}"
                selected_cam=${CAMERAS[0]}
            fi
        fi
        
        # Ask for verbose mode
        echo ""
        read -p "Enable verbose logging? [y/N]: " verbose_choice
        
        echo ""
        if [[ "$verbose_choice" =~ ^[Yy]$ ]]; then
            echo -e "${GREEN}▶${NC} Starting ChessDrum with camera ${YELLOW}${selected_cam}${NC} (${CYAN}VERBOSE${NC})..."
            echo ""
            python3 src/main.py --camera --cam "$selected_cam" --verbose "$@"
        else
            echo -e "${GREEN}▶${NC} Starting ChessDrum with camera ${YELLOW}${selected_cam}${NC}..."
            echo ""
            python3 src/main.py --camera --cam "$selected_cam" "$@"
        fi
    fi
    
else
    echo -e "${RED}✗${NC} Invalid choice. Exiting."
    exit 1
fi

# Exit message
echo ""
echo -e "${CYAN}═══════════════════════════════════════${NC}"
echo -e "${GREEN}✓${NC} ChessDrum closed"
echo -e "${CYAN}═══════════════════════════════════════${NC}"
