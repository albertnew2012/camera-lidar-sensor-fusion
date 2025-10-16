import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import cv2  # For loading the image with OpenCV

def project_lidar_to_image(points, img, intrinsics, extrinsics, size=1, color=(0, 0, 255)):
    """Project LiDAR points to a single image and draw circles on it."""
    if points is None or img is None:
        print("Warning: Missing points or image. Skipping projection.")
        return
    
    # LiDAR to image transformation
    lidar2img_rt = intrinsics @ extrinsics
    points= points[:,:3]  # Use only XYZ, ignore intensity if present
    points4d = np.hstack((points, np.ones((len(points), 1))))
    a = lidar2img_rt @ points4d.T
    z = a[2, :].clip(1e-6)  # Prevent division by zero
    u = (a[0, :] / z).astype(np.int32)
    v = (a[1, :] / z).astype(np.int32)
    h, w, _ = img.shape
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)
    u, v = u[valid].astype(np.int32), v[valid].astype(np.int32)
    # Draw blue points: loop over valid points since cv2.circle expects scalar centers
    for i in range(len(u)):
        cv2.circle(img, (u[i], v[i]), size, color, -1)  # Draw blue points in BGR

class SplitViewRealDataApp:
    def __init__(self, points, box_center, image1, image2):
        # Initialize the GUI app
        gui.Application.instance.initialize()
        
        # Create main window
        self.window = gui.Application.instance.create_window("Real Point Cloud + Images", 1280, 720)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_key(self._on_key)  # Add key event handler for quitting
        
        # Scene widget for 3D point cloud (add directly to window)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.window.add_child(self.scene_widget)
        
        # Image widget for first 2D image (add directly to window)
        self.image_widget1 = gui.ImageWidget()
        self.window.add_child(self.image_widget1)
        
        # Image widget for second 2D image (add directly to window)
        self.image_widget2 = gui.ImageWidget()
        self.window.add_child(self.image_widget2)
        
        # Load and display provided data
        self.add_point_cloud(points, box_center)
        self.add_image1(image1)
        self.add_image2(image2)

    def _on_key(self, event):
        # Handle 'Q' or 'q' key press to quit the application
        if event.type == gui.KeyEvent.Type.DOWN:
            if event.key == gui.KeyName.Q:  # Use KeyName.Q for the 'Q' key (covers both cases)
                gui.Application.instance.quit()

    def _on_layout(self, layout_context):
        # Get the content rectangle of the window
        r = self.window.content_rect
        
        # Position the scene widget on the left (e.g., 50% width, full height)
        left_width = int(r.width * 0.5)
        self.scene_widget.frame = gui.Rect(r.x, r.y, left_width, r.height)
        
        # Right side: full remaining width
        right_x = r.x + left_width
        right_width = r.width - left_width
        
        # Split right side vertically into two equal heights for the two images
        half_height = int(r.height * 0.5)
        self.image_widget1.frame = gui.Rect(right_x, r.y, right_width, half_height)
        self.image_widget2.frame = gui.Rect(right_x, r.y + half_height, right_width, half_height)

    def add_point_cloud(self, points, box_center):
        if points is None:
            print("Warning: No point cloud provided. Skipping.")
            return

        # --- main LiDAR points ---
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(np.ones_like(points[:, :3]) * 0.35)

        mat_points = rendering.MaterialRecord()
        mat_points.shader = "defaultUnlit"
        mat_points.point_size = 2.0  # <--- adjust size here

        self.scene_widget.scene.add_geometry("LiDARPoints", pcd, mat_points)

        # --- box_center points ---
        if box_center is not None and len(box_center) > 0:
            box_pcd = o3d.geometry.PointCloud()
            box_pcd.points = o3d.utility.Vector3dVector(box_center[:, :3])
            box_pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0, 1, 0]]), (len(box_center), 1)))  # green

            mat_box = rendering.MaterialRecord()
            mat_box.shader = "defaultUnlit"
            mat_box.point_size = 6.0  # <--- bigger box center points

            self.scene_widget.scene.add_geometry("BoxCenters", box_pcd, mat_box)
        
        # Initial camera setup based on bounds
        bounds = pcd.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())
        
        # Apply custom view control (translated to new API, with reversed direction)
        center = np.array([35, 0, 0])
        up = np.array([1, 0, 0])
        front = np.array([-6, 0, 1])  # adjust pitch angle by changing the first value
        front_norm = np.linalg.norm(front)
        front_unit = front / front_norm
        # Distance for zoom ~0.3 (zoomed out; adjust as needed for your data)
        distance = 45.0
        # Reversed: place eye on the opposite side for "near to far" view
        eye = center + distance * front_unit
        self.scene_widget.scene.camera.look_at(center, eye, up)

    def add_image1(self, image1):
        # Show provided Open3D image in first image widget
        if image1 is None:
            print("Warning: No image1 provided. Skipping.")
            return
        if isinstance(image1, np.ndarray):
            # If numpy array, convert to Open3D image (assuming BGR from cv2; convert to RGB if needed)
            if len(image1.shape) == 3 and image1.shape[2] == 3:
                image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            o3d_image = o3d.geometry.Image(image1)
        else:
            o3d_image = image1  # Assume already o3d.geometry.Image
        self.image_widget1.update_image(o3d_image)

    def add_image2(self, image2):
        # Show provided Open3D image in second image widget
        if image2 is None:
            print("Warning: No image2 provided. Skipping.")
            return
        if isinstance(image2, np.ndarray):
            # If numpy array, convert to Open3D image (assuming BGR from cv2; convert to RGB if needed)
            if len(image2.shape) == 3 and image2.shape[2] == 3:
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            o3d_image = o3d.geometry.Image(image2)
        else:
            o3d_image = image2  # Assume already o3d.geometry.Image
        self.image_widget2.update_image(o3d_image)

    def run(self):
        gui.Application.instance.run()

# --- Run the app ---
if __name__ == "__main__":
    # Load data externally and pass to the app
    # pcd_path = "/perception_data/lucid_data/batch_M1260_validation/data/3p9L06-testfleet-USA-vin123-20230811_152935/3p9L06-lidar-00-robosense-m1-20230811_152935/pcd_files/0000003639.pcd"
    # pcd = o3d.io.read_point_cloud(pcd_path)
    # if pcd is None:
    #     print(f"Warning: Failed to load {pcd_path}. Using empty point cloud.")
    #     pcd = o3d.geometry.PointCloud()  # Empty but valid
    
    # Create an empty Open3D PointCloud object (it will be populated later)
    # pcd = o3d.geometry.PointCloud()
    pcd_bin = "data/003679.bin"
    points = np.fromfile(pcd_bin, dtype=np.float32).reshape(-1, 4)
    points = points[~np.isnan(points).any(axis=1)]
    # pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Use only XYZ, ignore intensity
    img1_path = "data/003679.png"
    img1 = cv2.imread(img1_path)
    if img1 is None:
        print(f"Warning: Failed to load {img1_path}. Using blank image.")
    img2_path = "data/003679.png"
    img2 = cv2.imread(img2_path)
    if img2 is None:
        print(f"Warning: Failed to load {img2_path}. Using blank image.")
    
    # Camera intrinsics (normalized)
    intrinsics = np.array([[1.00925e+03, 0.00000e+00, 1.93636e+03, 0.00000e+00],
                                   [0.00000e+00, 1.00925e+03, 9.14244e+02, 0.00000e+00],
                                   [0.00000e+00, 0.00000e+00, 1.00000e+00, 0.00000e+00]])
    # intrinsics /= 2.0
    # intrinsics[-1, -1] = 1.0
    # LiDAR to image extrinsics
    extrinsics = np.array([[-0.00349035, -0.99990824, -0.0130896 , -0.03904478],
                                    [ 0.02098797, 0.01301354, -0.99969503, 0.58792128],
                                    [ 0.99977364, -0.00376401, 0.02094063, 1.7779269 ],
                                    [ 0. , 0. , 0. , 1. ]])
    
    # Project LiDAR points to img1 only (single image)
    box_center = np.array([[ 3.00288887e+01, -3.36531830e+00,  7.06310272e-02],
       [ 1.29292994e+01, -8.48119926e+00,  1.22069609e+00],
       [ 1.74340134e+01,  5.22304153e+00,  1.72252655e-01],
       [ 9.16942902e+01, -1.31328964e+00,  3.14395010e-01],
       [ 2.85593605e+01,  1.36676254e+01,  7.67679214e-01],
       [ 7.05907974e+01, -4.05837059e+00,  9.49960947e-03],
       [ 1.95270214e+01, -1.65919838e+01,  1.80955052e+00],
       [ 1.93684101e+01, -1.66631241e+01,  2.66315985e+00],
       [ 8.75024891e+00, -8.64268303e+00,  6.84491634e-01],
       [ 1.02026043e+01, -8.66271877e+00,  8.05699229e-01],
       [ 9.04570847e+01, -2.93526840e+00,  6.37429953e-01],
       [ 1.04898674e+02,  1.21621628e+01,  4.47237825e+00],
       [ 1.28137787e+02,  5.16837311e+00,  5.33331966e+00],
       [ 9.03447723e+01,  3.22303772e-02,  9.82473493e-01],
       [ 8.16676331e+01,  2.97419357e+00,  3.06258202e+00],
       [ 2.01024837e+01, -1.66695786e+01,  1.87374938e+00],
       [ 1.15018439e+00, -3.66300583e+00,  1.98135197e-01],
       [ 4.14524269e+01, -1.66567039e+01,  2.26372313e+00],
       [ 4.11835976e+01, -1.63415527e+01,  1.46880507e+00],
       [ 4.36748543e+01, -1.73081341e+01,  1.19536996e+00],
       [ 1.61497898e+01, -8.05065155e+00,  9.86256361e-01],
       [ 4.79831848e+01, -1.67322903e+01,  1.42130995e+00],
       [ 1.64646816e+01, -1.83446293e+01,  1.54363048e+00],
       [ 6.32051697e+01, -1.35669327e+01,  2.02562666e+00],
       [ 4.68521767e+01, -1.60576038e+01,  1.98052120e+00],
       [ 2.06908188e+01, -1.74149990e+01,  1.54523027e+00],
       [ 1.73982792e+01, -1.91545467e+01,  1.67425096e+00],
       [ 1.59544010e+01, -1.75148430e+01,  1.54384673e+00],
       [ 1.29417007e+02,  3.29697800e+00,  4.39307165e+00],
       [ 6.45239487e+01,  1.08329773e-01,  2.61669779e+00],
       [ 3.60995216e+01, -1.72952175e+01,  7.60565996e-01],
       [ 6.34600983e+01,  2.95600891e-02,  2.63639975e+00],
       [ 4.14239235e+01, -1.82770138e+01,  1.35937130e+00],
       [ 1.55646858e+01, -7.70780182e+00,  1.63992560e+00],
       [ 1.01687050e+02, -1.24480057e+00,  3.08383751e+00],
       [ 9.03439255e+01,  2.51485252e+00,  2.63241363e+00],
       [ 4.40403061e+01,  1.12921028e+01,  2.15506434e+00],
       [ 1.10494843e+02,  1.01638718e+01,  3.74649119e+00],
       [ 1.01138947e+02,  1.00769234e+00,  3.19902825e+00],
       [ 8.01289597e+01, -5.08004189e+00,  2.14529824e+00],
       [ 5.12327499e+01, -1.68837128e+01,  8.20940495e-01],
       [ 1.21774567e+02, -1.59015694e+01,  5.39886808e+00],
       [ 6.61351318e+01,  2.69972801e+00,  2.54541397e+00],
       [ 1.26240479e+02,  3.99555588e+00,  5.50798845e+00],
       [ 2.08778133e+01, -1.68111725e+01,  1.98693228e+00],
       [ 3.81730461e+01, -1.74696274e+01,  5.97090364e-01],
       [ 4.59941940e+01, -5.11235046e+00,  1.82430530e+00],
       [ 5.38980598e+01, -1.18679047e+00,  1.75961757e+00],
       [ 8.89707260e+01,  1.03728008e+01,  3.43679190e+00],
       [ 1.97103386e+01, -1.66240406e+01,  2.56925535e+00],
       [ 4.48446655e+01, -1.61330643e+01,  1.90383530e+00],
       [ 1.14249077e+02,  1.01673470e+01,  4.82730389e+00],
       [ 5.51252022e+01, -1.14381027e+00,  1.81572306e+00],
       [ 4.48045387e+01, -5.13908768e+00,  1.80181813e+00],
       [ 4.72269287e+01, -5.09501553e+00,  1.84921682e+00],
       [ 5.27291298e+01, -1.18672752e+00,  1.86600900e+00],
       [ 1.16220036e+01, -1.61618500e+01,  1.18435013e+00],
       [ 1.28711197e+02, -1.37936382e+01,  4.71187973e+00],
       [ 2.35699043e+01, -1.62954903e+01,  1.61057281e+00],
       [ 2.30735722e+01, -1.67703800e+01,  2.05447483e+00],
       [ 5.84083557e+01, -1.47694511e+01,  1.88035071e+00],
       [ 1.61688862e+01, -1.63386860e+01,  1.48656464e+00],
       [ 7.77526855e+01,  1.22607994e+01,  3.89218998e+00],
       [ 9.47238636e+00, -1.13350372e+01,  7.77372003e-01],
       [ 7.88509216e+01,  1.23710823e+01,  3.70975065e+00],
       [ 1.94483547e+01, -1.63858318e+01,  2.83739042e+00],
       [ 4.85575562e+01, -5.12824154e+00,  1.88612461e+00],
       [ 3.93046989e+01,  3.36462021e+00,  1.80894148e+00],
       [ 1.30686249e+02, -1.52053900e+01,  5.86813498e+00],
       [ 5.46042366e+01, -4.98772144e+00,  2.39773798e+00],
       [ 1.40719193e+02, -1.68766594e+01,  5.84396267e+00],
       [ 6.08174286e+01,  1.18629761e+01,  3.10373831e+00],
       [ 1.37038010e+02,  3.83443642e+00,  5.34304523e+00],
       [ 4.16771164e+01, -1.66088276e+01,  1.98765004e+00],
       [ 2.18896008e+01, -1.52855148e+01,  9.38723207e-01],
       [ 2.40746422e+01, -1.62792969e+01,  1.33696675e+00],
       [ 1.32715942e+02,  7.55813789e+00,  4.14606571e+00],
       [ 1.97391491e+01, -1.73918304e+01,  2.38469028e+00],
       [ 1.40756607e+02,  1.27086792e+01,  6.60359144e+00],
       [ 1.38877609e+02,  7.69026566e+00,  5.97448015e+00],
       [ 9.22819748e+01,  6.41352081e+00,  4.01609898e+00],
       [ 7.82425003e+01,  2.30680275e+00,  2.51210546e+00],
       [ 9.35433502e+01,  6.35890388e+00,  4.11241102e+00],
       [ 1.33881241e+02,  7.63413811e+00,  5.48225355e+00],
       [ 1.41436569e+02,  8.88940430e+00,  5.54975033e+00],
       [ 5.68590851e+01,  1.32492676e+01,  3.10644913e+00],
       [ 2.00007572e+01, -1.76322594e+01,  2.12435842e+00],
       [ 1.93407803e+01, -1.67014790e+01,  2.37260485e+00],
       [ 6.19740677e+01,  1.19144325e+01,  3.19761896e+00],
       [ 1.37660126e+02,  7.74212837e+00,  6.20151329e+00],
       [ 9.10211639e+01,  6.40239525e+00,  3.64226627e+00],
       [ 4.98489990e+01, -5.09356403e+00,  1.90116656e+00],
       [ 5.56368103e+01,  1.32491035e+01,  2.99311662e+00],
       [ 7.09807205e+01,  3.97057915e+00,  3.09530306e+00],
       [ 9.48271103e+01,  6.28075027e+00,  4.16464567e+00],
       [ 7.22235565e+01,  3.97654724e+00,  3.21017981e+00],
       [ 4.35859718e+01, -2.54382896e+00,  1.95234489e+00],
       [ 1.82998390e+01,  1.03408756e+01,  1.03744471e+00],
       [ 1.41439148e+02,  1.01536789e+01,  5.50206852e+00],
       [ 1.35202255e+02,  7.63683510e+00,  5.28733826e+00],
       [ 5.10587196e+01, -5.06932545e+00,  1.98938501e+00],
       [ 1.36393356e+02,  7.67709160e+00,  5.70440912e+00],
       [ 1.35740036e+02, -8.07643509e+00,  4.67916155e+00],
       [ 5.95616722e+01,  1.19185143e+01,  3.05679750e+00],
       [ 7.73066483e+01,  2.45717430e+00,  2.81736755e+00],
       [ 9.61254654e+01,  6.36020088e+00,  3.67108893e+00],
       [ 8.97851639e+01,  6.44832420e+00,  3.28536177e+00],
       [ 5.80677071e+01,  1.25576630e+01,  3.06830788e+00],
       [ 1.36324936e+02, -1.65974369e+01,  5.51968861e+00],
       [ 9.73770981e+01,  6.17866707e+00,  3.42874861e+00],
       [ 7.35688629e+01,  3.85156631e+00,  3.24285221e+00],
       [ 2.21573143e+01, -1.66588669e+01,  2.39030933e+00],
       [ 6.97504501e+01,  3.98168564e+00,  2.90618515e+00],
       [ 7.59269485e+01,  3.15917397e+00,  2.82382536e+00],
       [ 9.41504669e+01,  3.80470848e+00,  4.17093754e+00],
       [ 6.97561417e+01,  5.17567444e+00,  2.92483330e+00],
       [ 7.48575974e+01,  3.89175606e+00,  3.02664995e+00],
       [ 1.11077356e+01, -1.37842684e+01,  9.43078756e-01],
       [ 4.92664909e+01,  1.18632584e+01,  2.48982882e+00],
       [ 6.85921707e+01,  5.24489212e+00,  2.82084084e+00],
       [ 1.48346283e+02, -5.05168819e+00,  6.51052713e+00],
       [ 1.91592255e+01, -1.74671135e+01,  2.46960688e+00],
       [ 5.44209824e+01,  1.32823219e+01,  2.91865325e+00],
       [ 2.17317390e+01, -1.67167130e+01,  2.38042784e+00],
       [ 5.57420464e+01, -1.48093863e+01,  1.54859614e+00],
       [ 4.79963264e+01,  1.18776588e+01,  2.43612838e+00],
       [ 2.26904945e+01,  9.20178413e+00,  8.14931691e-01],
       [ 1.51340446e+01, -1.32164478e+01,  8.16199780e-01],
       [ 5.07063446e+01, -1.66072216e+01,  1.03850961e+00],
       [ 4.21071243e+01, -1.29360580e+00,  1.92137849e+00],
       [ 1.43213943e+02, -8.10626411e+00,  6.42049503e+00],
       [ 4.81987648e+01,  8.24810028e-01,  1.72055340e+00],
       [ 7.59150391e+01, -1.18005705e+01,  1.65390694e+00],
       [ 8.04222870e+01,  8.83913994e+00,  3.78921390e+00],
       [ 4.69654694e+01,  8.38787079e-01,  1.67015052e+00],
       [ 6.30092087e+01,  9.98550415e+00,  3.02747393e+00],
       [ 6.32402229e+01,  1.19346619e+01,  3.25942039e+00],
       [ 4.09738464e+01, -1.62057648e+01,  2.83544397e+00],
       [ 5.28126526e+01,  9.11434364e+00,  2.33973527e+00],
       [ 7.31655960e+01, -6.93037987e+00,  1.93062580e+00],
       [ 5.08638191e+01,  1.45705147e+01,  2.82979989e+00],
       [ 1.25002548e+02,  1.15132828e+01,  5.01311779e+00],
       [ 4.00110893e+01, -1.63918209e+01,  1.08992136e+00],
       [ 5.34417419e+01,  1.39171028e+01,  2.92602921e+00],
       [ 5.21789932e+01,  1.39144020e+01,  2.82826734e+00],
       [ 4.23985825e+01, -1.67261276e+01,  1.35077214e+00],
       [ 9.86493225e+01,  4.97832489e+00,  3.44462037e+00],
       [ 6.91756897e+01, -1.33739138e+01,  2.49436831e+00],
       [ 1.25026672e+02,  1.02344856e+01,  4.91852570e+00],
       [ 2.20679302e+01, -1.68357315e+01,  1.98954785e+00],
       [ 1.49049515e+02,  4.36259842e+00,  5.54844189e+00],
       [ 2.53569927e+01,  9.69531441e+00,  9.20889616e-01],
       [ 7.10965805e+01, -1.10524178e+00,  1.87738407e+00],
       [ 1.26284103e+02,  1.15456753e+01,  5.91204214e+00],
       [ 5.71187134e+01,  9.10631371e+00,  2.58435559e+00],
       [ 6.45126343e+01,  1.19664574e+01,  3.37336636e+00],
       [ 1.62999439e+01, -1.37804012e+01,  8.82556319e-01],
       [ 4.84825172e+01,  1.45874519e+01,  2.81255674e+00],
       [ 5.15951614e+01,  1.17760983e+01,  2.56423497e+00],
       [ 8.12594757e+01,  1.27421341e+01,  3.78499460e+00],
       [ 8.94890499e+00, -7.97100735e+00,  9.37395811e-01],
       [ 1.34440445e+02, -1.31397362e+01,  4.81515121e+00],
       [ 1.06710747e+02, -1.44735909e+00,  4.39334154e+00],
       [ 1.05406517e+02, -2.11867142e+00,  3.67734766e+00],
       [ 2.09562855e+01, -1.74020996e+01,  2.06118870e+00],
       [ 7.57646408e+01,  1.18381252e+01,  3.56644535e+00],
       [ 1.23752022e+02,  1.16264534e+01,  5.35604763e+00],
       [ 6.46354523e+01,  9.32305717e+00,  3.14689827e+00],
       [ 1.27690079e+02,  1.13599186e+01,  4.46588326e+00],
       [ 6.57914047e+01,  1.19684944e+01,  3.41511726e+00],
       [ 1.14948044e+02,  1.39844589e+01,  4.66132450e+00],
       [ 5.60262947e+01,  9.81856537e+00,  2.48441935e+00],
       [ 1.19985825e+02,  1.32924957e+01,  4.35575485e+00],
       [ 1.28806244e+02,  1.15111351e+01,  5.98787737e+00],
       [ 3.09629402e+01,  1.03022614e+01,  1.12698483e+00],
       [ 1.09795479e+02, -1.21888943e+01,  4.06676388e+00],
       [ 1.38906494e+02,  1.14687729e+01,  5.82223749e+00],
       [ 6.70122147e+01,  1.19769173e+01,  3.47619104e+00],
       [ 1.37677017e+02,  1.14973412e+01,  6.04561281e+00],
       [ 1.36411819e+02,  1.14500809e+01,  5.48236656e+00],
       [ 1.30197296e+02,  1.13882637e+01,  4.42699432e+00],
       [ 1.35161530e+02,  1.15325966e+01,  5.69485044e+00],
       [ 1.74665546e+01,  8.66504669e+00,  1.33845925e+00],
       [ 1.33859070e+02,  1.15375519e+01,  5.55989361e+00],
       [ 1.32710098e+02,  1.13883171e+01,  4.62178993e+00],
       [ 1.31325729e+02,  1.14702511e+01,  5.98516417e+00],
       [ 6.83212814e+01,  1.19579849e+01,  3.59995031e+00],
       [ 6.94826050e+01,  1.19700737e+01,  3.68738317e+00],
       [ 7.07358398e+01,  1.19269905e+01,  3.73120642e+00],
       [ 7.20452118e+01,  1.19141159e+01,  3.69581604e+00],
       [ 7.32676392e+01,  1.18551521e+01,  3.64028645e+00],
       [ 7.45937042e+01,  1.18926086e+01,  3.68971205e+00],
       [ 2.51918716e+01, -1.94377213e+01,  1.83790660e+00],
       [ 1.22519936e+02,  1.20651169e+01,  4.50610209e+00],
       [ 2.66177597e+01, -1.72260742e+01,  1.85078907e+00],
       [ 7.53494835e+00, -1.18013821e+01,  8.05096149e-01],
       [ 3.68751450e+01,  6.63759232e+00,  1.91830134e+00],
       [ 1.18747581e+02,  1.40118828e+01,  4.75939369e+00],
       [ 1.21268806e+02,  1.28001137e+01,  4.94005299e+00],
       [ 1.16216545e+02,  1.39872322e+01,  4.79051399e+00]])


    project_lidar_to_image(points, img1, intrinsics, extrinsics, size=1, color=(0, 0, 255))
    project_lidar_to_image(box_center, img2, intrinsics, extrinsics, size=3, color=(0, 255, 0))
    app = SplitViewRealDataApp(points,box_center, img1, img2)
    app.run()
    # for i in range(5):
    #     print(i)
    #     app = SplitViewRealDataApp(points, img1, img2)
        
    #     app.run()