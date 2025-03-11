from icecream import ic
from multiprocessing.pool import ThreadPool
import subprocess

# cf. https://stackoverflow.com/a/26783779/7272382

min_tree_boxes = 16
max_tree_boxes = 2048
num_sobol_samples = 64

def generate(slice_string):
    ic(slice_string)
    my_tool_subprocess = subprocess.Popen(
        "python3 ../thingies_with_omnitrees.py --sobol_samples={} --slice={} {}-{}".format(
            num_sobol_samples, slice_string, min_tree_boxes, max_tree_boxes
        ),
        shell=True,
        stdout=subprocess.PIPE,
    )
    my_tool_subprocess.wait()

def evaluate(slice_string, num_boxes):
    ic(slice_string, num_boxes)
    my_tool_subprocess = subprocess.Popen(
        "python3 ../thingies_with_omnitrees_evaluate.py --sobol_samples={} --slice={} {}".format(
            num_sobol_samples, slice_string, num_boxes
        ),
        shell=True,
        stdout=subprocess.PIPE,
    )
    my_tool_subprocess.wait()


num = 8  # set to the number of workers you want (None defaults to the cpu count of your machine)
tp = ThreadPool(num)

num_slices = 1 #2048
for sample in range(num_slices):
    tp.apply_async(generate, (str(sample) + "/" + str(num_slices),))
    # and/or:
    # num_boxes = min_tree_boxes
    # while num_boxes <= max_tree_boxes:
    #     tp.apply_async(evaluate, (str(sample) + "/" + str(num_slices), num_boxes))
    #     num_boxes *= 2

tp.close()
tp.join()
