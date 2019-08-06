import urllib.request
import os
import re

# ----- Configurations -----

# path to root ScanNet folder
scannet_dir = "/home/dtc/Data/ScanNet"

# --- end of Configurations ---

scannet_url = "http://kaldir.vc.in.tum.de/scannet/download-scannet.py"

downloader_path = os.path.join(scannet_dir, "download-scannet.py")
new_downloader = os.path.join(scannet_dir, "new-download-scannet.py")

# official downloader script
urllib.request.urlretrieve(scannet_url, downloader_path)

# Python 2 to Python 3
with open(downloader_path, "r") as f_in:
    with open(new_downloader, "w") as f_out:
        for line in f_in:
            if re.match(r"^import urllib\n$", line):
                f_out.write("# import urllib\n")
            elif re.match("^#import urllib.request \(for python3\)\n", line):
                f_out.write("import urllib.request\n")
            else:
                line = line.replace(r"raw_input(", "input(")
                line = line.replace(r"#scan_lines = urllib.request.urlopen(release_file)", "scan_lines = urllib.request.urlopen(release_file)")
                line = line.replace(r"scan_lines = urllib.urlopen(release_file)", "# scan_lines = urllib.urlopen(release_file)")
                line = line.replace(r"#urllib.request.urlretrieve(url, out_file_tmp)", "urllib.request.urlretrieve(url, out_file_tmp)")
                line = line.replace(r"urllib.urlretrieve(url, out_file_tmp)", "# urllib.urlretrieve(url, out_file_tmp)")
                f_out.write(line)

# download _vh_clean_2.ply
cmd_vh_clean_2 = "python " + new_downloader + " -o " + scannet_dir + " --type _vh_clean_2.ply"
os.system(cmd_vh_clean_2)
cmd_vh_clean_2_label = "python " + new_downloader + " -o " + scannet_dir + " --type _vh_clean_2.labels.ply"
os.system(cmd_vh_clean_2_label)
