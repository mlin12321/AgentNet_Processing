# import necessary modules
from pyspark import SparkConf, SparkContext
from system_prompts import L1_SYSTEM_PROMPT, L2_SYSTEM_PROMPT, L3_SYSTEM_PROMPT
import argparse

import os
os.environ['PYSPARK_PYTHON'] = '...'

SYSTEM_PROMPT_MAP = {
    "L1": L1_SYSTEM_PROMPT,
    "L2": L2_SYSTEM_PROMPT,
    "L3": L3_SYSTEM_PROMPT
}

def main(args):
    # create a SparkConf object
    conf = SparkConf() \
            .setAppName("MyApp") \
            .setMaster("local[*]") \
            .set("spark.driver.memory", "64g") \
            .set("spark.executor.memory", "24g") \
            .set("spark.executor.instances", "30") \
            .set("spark.driver.maxResultSize", "8g")

    # create a SparkContext object
    sc = SparkContext(conf=conf)

    # get the configuration settings
    current_conf = sc.getConf()

    # print the number of worker threadsis_
    num_workers = current_conf.get("spark.executor.instances")
    print("Number of worker threads: ", num_workers)

    # print other configuration settings
    print("Spark driver memory: ", current_conf.get("spark.driver.memory"))
    print("Spark executor memory: ", current_conf.get("spark.executor.memory"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="L2")
    args = parser.parse_args()
    main(args)


