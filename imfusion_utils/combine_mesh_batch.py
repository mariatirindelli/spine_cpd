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

    fid = open("combine_meshes.txt", "w")
    fid.write("VERT1;VERT2;VERT3;VERT4;VERT5;OUTDIR")
    if params.spine == "all":
        params.spine = ["spine" + str(item) for item in range(1, 22)]
    else:
        params.spine = [params.spine]

    for spine in params.spine:

        mesh_dir = os.path.join(params.root, spine, params.timestamp)
        if not os.path.exists(mesh_dir):
            continue
        fid.write("\n")
        vert_files = [item for item in os.listdir(mesh_dir) if "full" not in item and params.format in item]
        vert_files = order_by_vert(vert_files)

        for i, vert_file in enumerate(vert_files):
            vert_path = os.path.normpath(os.path.join(mesh_dir, vert_file))
            fid.write(str(vert_path) + ";")

        outdir = os.path.normpath(os.path.join(mesh_dir, spine + "_full" + args.timestamp.replace("ts", "") + params.format))
        fid.write(outdir)

    fid.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data generation testing')
    parser.add_argument('--root', type=str,
                        default="E:/NAS/jane_project/obj_files")
    parser.add_argument('--spine', type=str,
                        default="all")
    parser.add_argument('--timestamp', type=str,
                        default="ts_20_0")
    parser.add_argument('--format', type=str,
                        default=".obj")
    args = parser.parse_args()

    main(args)
