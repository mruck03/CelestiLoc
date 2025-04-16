from identify_stars import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u

# Constants
NUM_STARS = 1000
TRUE_LAT = np.deg2rad(43.7749)   # San Francisco latitude
TRUE_LON = np.deg2rad(-83.4194) # San Francisco longitude

K = np.array([
    [1, 0, 0],  # fx, 0, cx
    [0, 1, 0],  # 0, fy, cy
    [0, 0, 1]   # camera coordinates are normalized
])

def init_problem():
    # Generate synthetic star data (unit vectors on celestial sphere)
    np.random.seed(42)
    ra = np.random.uniform(200, 300, NUM_STARS)
    dec = np.random.uniform(-30, 60, NUM_STARS)  # Limit to visible part of sky

    # Add fake distances using parallax (~1kly to ~10kly)
    dists = np.random.uniform(1e19, 1e20, NUM_STARS)
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=dists*u.m, frame='icrs')

    # Convert to Cartesian (ICRS frame)
    cartesian = coords.cartesian.xyz.value.T  # shape (NUM_STARS, 3)

    # Simulate time of observation
    obs_time = Time("2025-04-14T04:00:00", format='isot', scale='utc')

    # Convert Earth location to ECEF
    earth_loc = EarthLocation(lat=TRUE_LAT*u.rad, lon=TRUE_LON*u.rad)
    obs_pos = np.array([earth_loc.x.value, earth_loc.y.value, earth_loc.z.value])  # meters

    # Simulate a true rotation (camera pointing up + slight yaw/pitch)
    true_rotation = R.from_euler('xyz', [10, 15, 0], degrees=True)
    camera_to_world = true_rotation.as_matrix()

    # Convert stars to camera frame (pretend camera is at Earth's surface)
    star_dirs = cartesian - obs_pos  # Vector from observer to star
    star_dirs /= np.linalg.norm(star_dirs, axis=1, keepdims=True)

    # Apply rotation to get what camera sees
    stars_in_camera = (camera_to_world.T @ star_dirs.T).T

    # Keep only stars in front of the camera (z > 0)
    visible = stars_in_camera[:, 2] > 0
    image_vectors = stars_in_camera[visible]

    # Project to image plane (focal length = 1, ignore K for toy)
    image_points = image_vectors[:, :2] / image_vectors[:, 2:]

    # Package results
    return image_points, cartesian[visible], TRUE_LAT, TRUE_LON, true_rotation.as_quat()


def main():
    #load problem
    image_points, star_points, true_lat, true_lon, true_quat = init_problem()

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5, origin=[0, 0, 0])

    distances = np.linalg.norm(star_points, axis=1)
    directions = star_points / distances[:, np.newaxis]
    new_distances = (distances / distances.max()) + 10
    star_points_moved = directions * new_distances[:, np.newaxis]
    pc_adj = star_points_moved

    pcd_adj = o3d.geometry.PointCloud()
    pcd_adj.points = o3d.utility.Vector3dVector(pc_adj)
    pcd_adj.colors = o3d.utility.Vector3dVector(np.ones((len(pc_adj), 3)) * [0, 0, 1])

    earth = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)

    t = lla_to_ecef(TRUE_LAT, TRUE_LON)

    print("True Quaternion:", true_quat)
    print("True Position (Lat, Lon):", np.degrees(np.array([TRUE_LAT, TRUE_LON])))

    T = quaternion_to_transformation_matrix(true_quat, t/np.linalg.norm(t))
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.7)
    frame.transform(T)

    quat = solve_rotation_svd(image_points, star_points, K, t)
    quat = quat / np.linalg.norm(quat)

    T = quaternion_to_transformation_matrix(quat, t/np.linalg.norm(t))
    pred_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.7)
    pred_frame.transform(T)

    # Create coordinate frame (origin)
    o3d.visualization.draw_geometries([frame, pred_frame, axis, pcd_adj, earth])

    N = 500

    pf = ParticleFilter(
        num_particles=N,
        image_points=image_points,
        star_points=star_points,
        camera_matrix=K
    )

    # Run particle filter for a fixed number of iterations
    initial_lat_std = 1
    initial_lon_std = 1
    num_iters = 51
    num_stars = 50

    for i in tqdm(range(num_iters)):
        lat_std = initial_lat_std
        lon_std = initial_lon_std
        pf.predict(np.deg2rad(lat_std), np.deg2rad(lon_std), image_points, star_points, K)

        #subsample a random amount of stars (num_stars). Stars are weighted by distance
        distances = np.linalg.norm(star_points, axis=1)
        weights = 1 / (distances + 1e-8)
        weights /= weights.sum()
        idx_stars = np.random.choice(len(star_points), num_stars, replace=False, p = weights)

        # pf.update_weights(image_points[idx_stars], star_points[idx_stars], K, scale=5)
        pf.update_weights(image_points, star_points, K, scale=5)
        
        #Print to see it change over time
        lat, lon = pf.estimate()
        # print("Estimated Position (Lat, Lon):", np.degrees(np.array([lat, lon])))
        # print(len(pf.particles))

        if i % 50 == 0:
            visualize_particles(pf.particles, step=i, estimate=(lat, lon))
            # frames = []
            # for p in pf.particles:
            #     t = lla_to_ecef(p.lat, p.lon)
            #     T = quaternion_to_transformation_matrix(p.quat, t/np.linalg.norm(t))
            #     pred_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.7)
            #     pred_frame.transform(T)
            #     frames.append(pred_frame)
            # o3d.visualization.draw_geometries(frames + [axis, pcd_adj, earth])



        print(N*0.7)
        print(pf.N_eff())

        if pf.N_eff() < N*0.7:
            print("resampling!!!!")
            pf.resample()

    # Final pose estimate
    lat, lon = pf.estimate()
    t = lla_to_ecef(lat, lon)

    quat = solve_rotation_svd(image_points, star_points, K, t)
    quat = quat / np.linalg.norm(quat)


    

    print("Estimated Quaternion:", quat)
    print("Estimated Position (Lat, Lon):", np.degrees(np.array([lat, lon])))

    T = quaternion_to_transformation_matrix(quat, t/np.linalg.norm(t))

    print("Transformation Matrix")
    print(T)

    #Vizualize
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.7)
    frame.transform(T)

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5, origin=[0, 0, 0])


    distances = np.linalg.norm(star_points, axis=1)
    # print(distances)

    # 2. Get normalized directions
    directions = star_points / distances[:, np.newaxis]
    # print(directions)

    # 3. Increase each distance by 10
    new_distances = (distances / distances.max()) + 10

    # 4. Scale direction vectors by new distances
    star_points_moved = directions * new_distances[:, np.newaxis]

    # pc = np.vstack((x_norm, y_norm, z_norm)).T
    pc_adj = star_points_moved
    # print(pc.shape)

    # Create Open3D point cloud
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc)
    # pcd.colors = o3d.utility.Vector3dVector(np.ones((len(pc), 3)) * [1, 0, 0])

    pcd_adj = o3d.geometry.PointCloud()
    pcd_adj.points = o3d.utility.Vector3dVector(pc_adj)
    pcd_adj.colors = o3d.utility.Vector3dVector(np.ones((len(pc_adj), 3)) * [0, 0, 1])

    earth = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    

    # Create coordinate frame (origin)
    o3d.visualization.draw_geometries([frame, axis, pcd_adj, earth])


if __name__ == "__main__":
    main()