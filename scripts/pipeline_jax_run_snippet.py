    def run(self, image_folder, output_path="output/reconstruction"):
        os.makedirs(output_path, exist_ok=True)
        img_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.jpg') or f.lower().endswith('.png')])
        
        # Limit to 5 frames for test
        img_files = img_files[:5]
        
        all_points = []
        all_colors = []
        
        # Global transformation (relative to first frame)
        T_global = jnp.eye(4)
        poses = {img_files[0]: np.array(T_global)}
        
        prev_data = self.process_image(os.path.join(image_folder, img_files[0]))
        
        for i in tqdm(range(1, len(img_files)), desc="Reconstructing"):
            img_name = img_files[i]
            curr_data = self.process_image(os.path.join(image_folder, img_name))
            
            # Get Top Keypoints
            def get_kpts_desc(scores, desc, k=1024):
                indices = jnp.argsort(scores.flatten())[::-1][:k]
                y, x = jnp.unravel_index(indices, scores.shape)
                kpts = jnp.stack([x, y], axis=-1)
                iy = (y / 8).astype(jnp.int32)
                ix = (x / 8).astype(jnp.int32)
                sampled_desc = desc[iy, ix, :]
                return kpts, sampled_desc

            kpts0, desc0 = get_kpts_desc(prev_data['sp_scores'], prev_data['sp_desc'])
            kpts1, desc1 = get_kpts_desc(curr_data['sp_scores'], curr_data['sp_desc'])
            
            # Step B: LightGlue Matching
            lg_input = {
                "image0": {"keypoints": kpts0[None], "descriptors": desc0[None]},
                "image1": {"keypoints": kpts1[None], "descriptors": desc1[None]}
            }
            lg_out = self.jit_lg(self.variables['lg'], lg_input)
            
            # Filter matches
            scores = lg_out['scores'][0, :-1, :-1]
            m0 = jnp.argmax(scores, axis=1)
            m1 = jnp.argmax(scores, axis=0)
            mutual = (jnp.arange(len(m0)) == m1[m0])
            valid = mutual & (jnp.exp(jnp.max(scores, axis=1)) > 0.1)
            
            idx0 = jnp.where(valid)[0]
            idx1 = m0[idx0]
            
            matched_kpts0 = kpts0[idx0]
            matched_kpts1 = kpts1[idx1]
            
            if len(idx0) > 8:
                # Step C: 3D Registration
                p0_3d = lift_points(matched_kpts0, prev_data['inv_depth'], prev_data['fov'])
                p1_3d = lift_points(matched_kpts1, curr_data['inv_depth'], curr_data['fov'])
                
                R, t = kabsch_alignment(p0_3d, p1_3d)
                
                T_curr = jnp.eye(4).at[:3, :3].set(R).at[:3, 3].set(t)
                # T_global maps points from Frame 0 to Global. 
                # Frame N = T_curr * Frame N-1
                # To bring Frame N to Global: Global = T_accum * Frame N
                # T_accum_N = T_accum_N-1 * T_curr_inv?
                # Actually, if Frame N-1 is already in Global, and we have N relative to N-1
                T_global = T_global @ jnp.linalg.inv(T_curr)
                
            poses[img_name] = np.array(T_global)

            # Store some points for visualization (Step D)
            h_step, w_step = 8, 8
            y_grid, x_grid = jnp.mgrid[0:768:h_step, 0:768:w_step]
            kpts_all = jnp.stack([x_grid.flatten(), y_grid.flatten()], axis=-1)
            # Need to scale grid to depth map size (1536)
            kpts_all_scaled = kpts_all * 2.0
            p_3d_all = lift_points(kpts_all_scaled, curr_data['inv_depth'], curr_data['fov'])
            p_global = apply_transform(p_3d_all, T_global[:3, :3], T_global[:3, 3])
            
            rgb_1536 = cv2.resize(curr_data['img_rgb'], (1536, 1536))
            colors = rgb_1536[::h_step*2, ::w_step*2, :].reshape(-1, 3) / 255.0
            
            all_points.append(p_global)
            all_colors.append(colors)
            
            prev_data = curr_data
            
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)
        o3d.io.write_point_cloud(os.path.join(output_path, "point_cloud.ply"), pcd)
        
        # Save poses for validation
        np.save(os.path.join(output_path, "poses.npy"), poses)
        
        print("Generating mesh...")
        pcd.estimate_normals()
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        o3d.io.write_triangle_mesh(os.path.join(output_path, "mesh.glb"), mesh)
        
        print(f"Reconstruction saved to {output_path}")
        return poses
