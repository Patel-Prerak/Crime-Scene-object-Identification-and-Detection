
import os
import numpy as np

def create_aframe_scene(detections, image_path, depth_path, depth_array, img_w, img_h, output_file):
    """
    Generates a Full 3D A-Frame scene using Displacement Maps.
    
    Args:
        detections: List of dicts with Box, Label, Conf.
        image_path: Path/DataURI to the RGB image.
        depth_path: Path/DataURI to the Depth map image.
        depth_array: Numpy array of depth values (normalized 0-255 or 0-1).
        img_w, img_h: Dimensions.
        output_file: Output path.
    """
    
    # 3D World Config
    WALL_Z = -4.0 # Moved back slightly to accommodate larger size
    WALL_WIDTH = 12.0 # Increased from 4.0 to 12.0
    aspect_ratio = img_h / img_w
    WALL_HEIGHT = WALL_WIDTH * aspect_ratio
    
    # Center of wall
    WALL_X = 0
    WALL_Y = 1.6
    
    X_min = WALL_X - (WALL_WIDTH / 2)
    Y_max = WALL_Y + (WALL_HEIGHT / 2)
    
    # Displacement Config
    DISPLACEMENT_SCALE = 3.5 # Increased from 1.5 to match scale
    
    scene_objects = []
    
    # Resolve Paths
    if image_path.startswith("data:"): rel_image_path = image_path
    else: rel_image_path = os.path.basename(image_path)
        
    if depth_path.startswith("data:"): rel_depth_path = depth_path
    else: rel_depth_path = os.path.basename(depth_path)
    
    # --- PROJECTION LOGIC ---
    # The depth map in A-Frame pushes geometry along the normal (Z+ relative to plane)
    # The plane is at WALL_Z.
    # Displacement moves vertices towards camera (if facing camera).
    # New Z = WALL_Z + (PixelValueNormalized * DISPLACEMENT_SCALE)
    
    # Add the 3D Plane
    # segments-width/height determines the mesh resolution (poly count)
    scene_objects.append(f"""
        <a-plane 
            src="{rel_image_path}" 
            displacement-map="{rel_depth_path}"
            displacement-scale="{DISPLACEMENT_SCALE}"
            displacement-bias="0"
            position="{WALL_X} {WALL_Y} {WALL_Z}" 
            width="{WALL_WIDTH}" 
            height="{WALL_HEIGHT}"
            segments-width="128" 
            segments-height="128"
            material="shader: standard; roughness: 1; metalness: 0; side: double">
        </a-plane>
    """)
    
    # Normalize depth array to 0-1 for calculation if it isn't already
    if depth_array.max() > 1.0:
        d_min = depth_array.min()
        d_max = depth_array.max()
        if d_max - d_min == 0:
            norm_depth = np.zeros_like(depth_array, dtype=float)
        else:
            norm_depth = (depth_array - d_min) / (d_max - d_min)
    else:
        norm_depth = depth_array

    # Store placed label positions to check for collisions: (x, y, z)
    placed_labels = []
    LABEL_HEIGHT = 0.3 # Estimated height of a label in 3D units
    MIN_DIST = 0.4 # Minimum distance between labels

    # Sort detections by Y (bottom to top) so we stack upwards? 
    # Or just process them order. Let's process in order but check collisions.
    
    for det in detections:
        label = det['Label']
        conf = det['Conf']
        x1, y1, x2, y2 = det['Box']
        
        # 1. Calculate XY on the 2D Plane
        nx1, ny1 = x1 / img_w, y1 / img_h
        nx2, ny2 = x2 / img_w, y2 / img_h
        
        ncx = (nx1 + nx2) / 2
        ncy = (ny1 + ny2) / 2
        nw = nx2 - nx1
        nh = ny2 - ny1
        
        cx3d_base = X_min + (ncx * WALL_WIDTH)
        cy3d_base = Y_max - (ncy * WALL_HEIGHT) 
        
        w3d = nw * WALL_WIDTH
        h3d = nh * WALL_HEIGHT
        
        # 2. Calculate Z-Depth from Depth Map
        py = int(ncy * (img_h - 1))
        px = int(ncx * (img_w - 1))
        
        try:
            depth_val = norm_depth[py, px]
        except:
            depth_val = 0.5 
            
        z_pos = WALL_Z + (depth_val * DISPLACEMENT_SCALE) + 0.1
        
        color = "red"
        if "Person" in label: color = "#fbbf24" # yellow
        if "Gun" in label: color = "#ef4444" # red
        if "Knife" in label: color = "#3b82f6" # blue
        
        # --- COLLISION AVOIDANCE ---
        # Initial proposed label position (relative to the object center)
        # Global position would be: cx3d_base, cy3d_base + h3d/2 + 0.2, z_pos
        base_label_y = cy3d_base + (h3d / 2) + 0.2
        
        # Check against existing labels
        adjusted_y = base_label_y
        
        collision = True
        while collision:
            collision = False
            for (lx, ly, lz) in placed_labels:
                # Check 2D distance in the viewing plane (X/Y)
                # We mainly care if they overlap in X and Y.
                dist_x = abs(cx3d_base - lx)
                dist_y = abs(adjusted_y - ly)
                
                # If they are close horizontally AND vertically
                if dist_x < 0.5 and dist_y < LABEL_HEIGHT:
                    collision = True
                    adjusted_y += LABEL_HEIGHT # Move up
                    break
        
        placed_labels.append((cx3d_base, adjusted_y, z_pos))
        
        # Calculate local offset for the entity relative to the parent position
        # Parent is at {cx3d_base} {cy3d_base} {z_pos}
        # Label needs to be at global Y = adjusted_y
        # So local Y = adjusted_y - cy3d_base
        local_label_y = adjusted_y - cy3d_base

        scene_objects.append(f"""
        <a-entity position="{cx3d_base} {cy3d_base} {z_pos}">
            <!-- Wireframe Box -->
            <a-box width="{w3d}" height="{h3d}" depth="0.1" 
                   material="color: {color}; wireframe: true">
            </a-box>
            
            <!-- Floating Label -->
            <!-- Use collision-adjusted Y position -->
            <a-entity position="0 {local_label_y} 0" look-at="[camera]">
                <a-text value="{label}" color="white" align="center" width="4" shader="msdf" font="https://raw.githubusercontent.com/etiennepinchon/aframe-fonts/master/fonts/orbitron/Orbitron-Black.json"></a-text>
                <a-text value="{conf:.0%}" position="0 -0.15 0" color="#ddd" align="center" width="2.5"></a-text>
                <!-- Connecting Line if moved far -->
                <a-entity line="start: 0 {-local_label_y + h3d/2} 0; end: 0 -0.25 0; color: {color}; opacity: 0.5"></a-entity>
            </a-entity>
        </a-entity>
        """)

    html_content = f"""<!DOCTYPE html>
<html>
  <head>
    <title>Full 3D Evidence Scene</title>
    <script src="https://aframe.io/releases/1.4.0/aframe.min.js"></script>
    <script src="https://unpkg.com/aframe-look-at-component@0.8.0/dist/aframe-look-at-component.min.js"></script>
  </head>
  <body>
    <a-scene background="color: #050505" fog="type: exponential; color: #000; density: 0.05">
      
      <!-- Ambient Light -->
      <a-light type="ambient" color="#222"></a-light>
      
      <!-- Spotlight on the evidence wall -->
      <a-light type="spot" position="0 4 2" target="#evidence-wall" color="#a855f7" intensity="0.8" angle="60" penumbra="0.5"></a-light>
      <a-light type="point" position="2 2 2" intensity="0.4" color="#fff"></a-light>

      <!-- Player Rig -->
      <a-entity id="rig" position="0 1.6 2">
        <a-camera look-controls wasd-controls="acceleration: 20">
            <a-cursor color="#a855f7" scale="0.5 0.5 0.5"></a-cursor>
        </a-camera>
      </a-entity>
      
      <!-- The Generated 3D Scene is injected here -->
      {''.join(scene_objects)}
      
      <!-- Environment Context -->
      <a-grid helper geometry="primitive: plane; width: 100; height: 100" rotation="-90 0 0" 
              material="src: url(https://cdn.aframe.io/a-painter/images/floor.jpg); repeat: 50 50; metalness: 0.6; roughness: 0.4; color: #333"></a-grid>
    
    </a-scene>
  </body>
</html>
"""
    
    with open(output_file, "w", encoding='utf-8') as f:
        f.write(html_content)
    
    return output_file
