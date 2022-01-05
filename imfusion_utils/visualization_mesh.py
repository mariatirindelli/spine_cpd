import argparse
import os


def order_by_vert(vert_list):
    ordered_files = [ [item for item in vert_list if "vert1" in item][0],
                      [item for item in vert_list if "vert2" in item][0],
                      [item for item in vert_list if "vert3" in item][0],
                      [item for item in vert_list if "vert4" in item][0],
                      [item for item in vert_list if "vert5" in item][0]]

    return ordered_files


def main(params):

    results_mesh_dir = os.path.join(params.result_folder, params.spine, params.iteration)
    print(results_mesh_dir)
    if not os.path.exists(results_mesh_dir):
        return

    fid = open("visualize_results.txt", "w")
    fid.write("VERT1;VERT2;VERT3;VERT4;VERT5;SOURCEDIR;TARGETDIR;USDIR")

    fid.write("\n")
    vert_files = [item for item in os.listdir(results_mesh_dir) if params.format in item]
    vert_files = order_by_vert(vert_files)

    ts_target = "ts_20_0" if params.spine != "spine5" and params.spine != "spine6" else "ts_19_0"

    for i, vert_file in enumerate(vert_files):
        vert_path = os.path.normpath(os.path.join(results_mesh_dir, vert_file))
        fid.write(str(vert_path) + ";")

    source_file = [item for item in os.listdir(os.path.join(params.gt_folder,
                                                            "obj_files",
                                                            params.spine,
                                                            "ts0")) if "full" in item][0]

    target_file = [item for item in os.listdir(os.path.join(params.gt_folder,
                                                            "obj_files",
                                                            params.spine,
                                                            ts_target)) if "full" in item][0]

    source_dir = os.path.normpath(os.path.join(params.gt_folder, "obj_files", params.spine, "ts0", source_file))
    target_dir = os.path.normpath(os.path.join(params.gt_folder, "obj_files", params.spine, ts_target, target_file))

    us_dir = os.path.join(params.gt_folder, "simulated_us", params.spine, ts_target + ".imf")

    fid.write(source_dir + ";" + target_dir + ";" + os.path.normpath(us_dir))

    fid.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data generation testing')
    parser.add_argument('--gt_folder', type=str,
                        default="E:/NAS/jane_project")
    parser.add_argument('--result_folder', type=str)
    parser.add_argument('--spine', type=str)
    parser.add_argument('--iteration', type=str)
    parser.add_argument('--format', type=str,
                        default=".obj")
    args = parser.parse_args()

    main(args)
