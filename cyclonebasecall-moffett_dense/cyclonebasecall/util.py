import os
from shutil import rmtree
from zipfile import ZipFile
import requests
from tqdm import tqdm
from glob import glob
from pathlib import Path
from itertools import chain
from functools import partial
from multiprocessing import Pool
from cycloneio.bgi_file import BGIFile

class DownloadZipFile:
    def __init__(self, url, zip_file_name, des_dir, force=False):
        self.file_name = zip_file_name
        self.data_name = zip_file_name.strip(".zip")
        self.url = "{}/{}".format(url, zip_file_name)
        self.force = force
        self.des_dir = des_dir

    def zip_file_path(self):
        return os.path.join(self.des_dir, self.file_name)

    def data_folder_dir(self):
        return os.path.join(self.des_dir, self.data_name)

    def download(self):
        """
        Download the remote file
        """
        # create the requests for the file
        req = requests.get(self.url, stream=True)
        total = int(req.headers.get('content-length', 0))

        # skip download if local file is found
        if os.path.exists(self.data_folder_dir()) and not self.force:
            print("[skipping {}]".format(self.data_folder_dir()))
            return

        if os.path.exists(self.data_folder_dir()) and self.force:
            if os.path.isfile(self.data_folder_dir()):
                os.remove(self.data_folder_dir())
            else:
                rmtree(self.data_folder_dir())

        # download zip file
        with tqdm(total=total, unit='iB', ascii=True, ncols=100, unit_scale=True, leave=False) as t:
            with open(self.zip_file_path(), 'wb') as f:
                for data in req.iter_content(1024):
                    f.write(data)
                    t.update(len(data))

        print("[downloaded: {}]".format(self.zip_file_path()))

        # unzip .zip files
        if self.file_name.endswith('.zip'):
            with ZipFile(self.zip_file_path(), 'r') as zfile:
                zfile.extractall(self.des_dir)
            os.remove(self.zip_file_path())


models = {
    "dna_r10.4.1_e8.2_fast@v3.5.1": "czyxdmbm56mt7lcgsinre7le70vqxtnf.zip",
    "dna_r10.4.1_e8.2_hac@v3.5.1": "per9xu2sqd1qjxqika6el6xh91pcyhd2.zip",
    "dna_r10.4.1_e8.2_sup@v3.5.1": "wpqoxvfa8mrg0kledy08vgmwwqq3m8ba.zip",

    "dna_r10.4_e8.1_sup@v3.4": "q07kb0p2f3qbj8cpnb35l1tcink21k82.zip",
    "dna_r10.4_e8.1_hac@v3.4": "tlxt45rmp5sv2c0f0p3jy8qf24lsidgv.zip",
    "dna_r10.4_e8.1_fast@v3.4": "biesc8w0ug1d59gpqpjin1utgcevvhs7.zip",

    "dna_r9.4.1_e8.1_sup@v3.3": "ts3h4rv1hqhjcv7t6wwng1dcfpzujvkg.zip",
    "dna_r9.4.1_e8.1_hac@v3.3": "25gua32bys32gnvj240hsml2ixnh4drc.zip",
    "dna_r9.4.1_e8.1_fast@v3.4": "h55evnpfq041ghniapmbklzqkdohkmo5.zip",

    "dna_r9.4.1_e8_sup@v3.3": "w10qhiggmg32gjcag6dv1wmwipzmcj84.zip",
    "dna_r9.4.1_e8_hac@v3.3": "5evhvoqb07u6d3y4jfy2oi3bmyrww293.zip",
    "dna_r9.4.1_e8_fast@v3.4": "3bq726s52at88zd9ve53x4px4quf3dpg.zip",
}


training = [
    "cmh91cxupa0are1kc3z9aok425m75vrb.hdf5",
]


def f_get_read_ids(
    file_path, read_ids=None, skip=False, auto_trim_adaptor = True, is_trim_adaptor = True, 
    min_orig_read_len = 10010, max_orig_read_len = 10000000,
    min_trim_adaptor_read_len = 100, max_trim_adaptor_read_len = 10000000):
    """
    Get match conditions read_ids.
    """
    ids = []
    try:
        with BGIFile(file_path, 'r', auto_trim_adaptor = auto_trim_adaptor, is_trim_adaptor = is_trim_adaptor) as f5_fh:
            for read in f5_fh.get_ori_reads():
                if is_trim_adaptor:
                    if (min_orig_read_len < read.orig_signal_len < max_orig_read_len) and (min_trim_adaptor_read_len < read.signal_len < max_trim_adaptor_read_len):
                        yield [file_path, read.read_id, auto_trim_adaptor, is_trim_adaptor]
                else:
                    if min_orig_read_len < read.orig_signal_len < max_orig_read_len:
                        yield [file_path, read.read_id, auto_trim_adaptor, is_trim_adaptor]
    except:
        raise IOError(f'loading {file_path} raise error!')  
    # if read_ids is None:
    #     return ids
    # return [rid for rid in ids if (rid[1] in read_ids) ^ skip]


def f_get_raw_data_for_read(info):
    """
    Get the raw signal from the ccf file for a given file_path, read_id pair
    """
    file_path, read_id, auto_trim_adaptor, is_trim_adaptor = info
    try:
        with BGIFile(file_path, 'r', auto_trim_adaptor = auto_trim_adaptor, is_trim_adaptor = is_trim_adaptor) as f5_fh:
            return f5_fh.get_read(read_id)
    except:
        raise IOError(f'loading {file_path} {read_id} raise error!')


def get_bgi_reads(
    reads_path, read_ids=None, skip=False, n_proc=1, recursive=False, cancel=None, auto_trim_adaptor = True, is_trim_adaptor = True, 
    min_orig_read_len = 10010, max_orig_read_len = 10000000,
    min_trim_adaptor_read_len = 100, max_trim_adaptor_read_len = 10000000):
    """
    Get all reads in a given `directory`.
    """
    pattern = "**/*.[c,f][c,a][f,s]**" if recursive else "*.[c,f][c,a][f,s]**"
    f_get_read_ids2 = partial(f_get_read_ids, read_ids=read_ids, skip=skip, auto_trim_adaptor = auto_trim_adaptor, is_trim_adaptor = is_trim_adaptor, min_orig_read_len = min_orig_read_len, max_orig_read_len = max_orig_read_len, min_trim_adaptor_read_len = min_trim_adaptor_read_len, max_trim_adaptor_read_len = max_trim_adaptor_read_len)
    if os.path.isfile(reads_path):
        reads = (reads_path, )
    else:
        reads = glob(reads_path + "/" + pattern, recursive=True)
    
    is_err_file = False
    try:
        for p_i in reads:
            # with BGIFile(p_i, 'r') as f5_fh:
            #     len(f5_fh.grp1.keys())
            pass
    except:
        is_err_file = True

    for job in chain(map(f_get_read_ids2, reads)):
        for read in map(f_get_raw_data_for_read, job):
            yield read
            if cancel is not None and cancel.is_set():
                return
