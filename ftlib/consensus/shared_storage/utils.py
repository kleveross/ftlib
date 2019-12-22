import logging
import os


def try_write_file(directory, filename, content):
    logging.info("writing: {}/{}".format(directory, filename))
    with open(os.path.join(directory, filename), "w") as f:
        try:
            f.write(content)
        except Exception as e:
            logging.warning("Error!" + str(e))
        else:
            return True
        return False


def list_only_files(directory):
    fs = []
    for (_, _, filenames) in os.walk(directory):
        fs.extend(filenames)
        break
    return [f for f in fs if f.count(".") == 3]


def filter_ip_files(filename_list):
    return [filename for filename in filename_list if filename.count(".") == 3]


def ip_to_int(s):
    return int(s.replace(".", ""))


class IOTool:
    def __init__(self, path):
        self._path = path

    def register_ip(self, ip, count):
        try_write_file(self._path, ip, str(count))

    def retrieve_ip(self, my_ip):
        logging.debug("start to retrieve ip")
        original_ip_files = list_only_files(self._path)
        ip_files = filter_ip_files(original_ip_files)
        if my_ip in ip_files:
            ip_files.remove(my_ip)
        counts = [
            open(os.path.join(self._path, f)).read().replace("\n", "")
            for f in ip_files
        ]
        try:
            counts = [int(c) for c in counts]
        except ValueError as e:
            logging.error("error when retrieve ip " + str(e))
        else:
            logging.debug(
                "ip retrieved " + str(ip_files) + " " + str(counts) + " alone"
                if len(ip_files) == 0
                else " not alone"
            )
            return ip_files, counts, len(ip_files) == 0
