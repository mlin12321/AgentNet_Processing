# import necessary modules
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from system_prompts import L1_SYSTEM_PROMPT, L2_SYSTEM_PROMPT, L3_SYSTEM_PROMPT
import argparse
import json

# import os
# os.environ['PYSPARK_PYTHON'] = '...'

SYSTEM_PROMPT_MAP = {
    "L1": L1_SYSTEM_PROMPT,
    "L2": L2_SYSTEM_PROMPT,
    "L3": L3_SYSTEM_PROMPT
}

def convert_to_openai_format(row, reason_type):
    all_data_points = []
    
    for i in range(1, len(row["traj"])):
        # Create a new messages list for each trajectory prefix
        messages = [
            {
                "role" : "system",
                "content" : SYSTEM_PROMPT_MAP[reason_type]
            }
        ]
        
        partial_traj = row["traj"][:i + 1]
        for j in range(len(partial_traj)):
            action_step = partial_traj[j].value
            format_assistant_content = ["# Step " + str(partial_traj[j]["index"] + 1)]

            # Dpepending on what kind of data is passed in, we include auxiliary types of reasoning
            # However, this is only for the last step
            if j == len(partial_traj) - 1:
                # L3 always gets observation + Thought
                # L2 gets Thought
                # L1 doesn't get anything
                # Not sure where reflection data goes?
                if reason_type == "L3":
                    format_assistant_content.append("## Observation: " + action_step.observation + "\n")
                if reason_type == "L2" or reason_type == "L3":
                    format_assistant_content.append("## Thought: " + action_step.thought + "\n")
                if reason_type == "L1":
                    pass

            # All reasoning types get the action
            format_assistant_content.append("## Action: " + action_step.action)

            # If it's the last step, we append the action code 
            if j == len(partial_traj) - 1:
                format_assistant_content.append("\n## Code: ```python\\n" + action_step.code + "```")

            messages.append({
                "role" : "assistant",
                "content" : "\n".join(format_assistant_content)
            })
    
        
            messages.append({
                "role" : "user",
                "image" : partial_traj[j]["image"]
            })
        
        # Add this trajectory prefix as a separate data point
        all_data_points.append(messages)

    return all_data_points

def main(args):
    # create a SparkConf object
    conf = SparkConf() \
            .setAppName("MyApp") \
            .setMaster("local[*]") \
            .set("spark.driver.memory", "64g") \
            .set("spark.executor.memory", "24g") \
            .set("spark.driver.maxResultSize", "8g")

    # create a SparkSession
    spark = SparkSession.builder.config(conf=conf).appName("ReadJSON").getOrCreate()
    
    # create a SparkContext object
    sc = spark.sparkContext
    current_conf = sc.getConf()
    
    # print the number of worker threadsis_
    num_workers = current_conf.get("spark.executor.instances")
    print("Number of worker threads: ", num_workers)

    # print other configuration settings
    print("Spark driver memory: ", current_conf.get("spark.driver.memory"))
    print("Spark executor memory: ", current_conf.get("spark.executor.memory"))

    # read a JSON file into a DataFrame
    df = spark.read.json("agentnet_ubuntu_5k.jsonl")

    # # show the first 10 rows of the DataFrame
    # df.show(10)

    # Map the data to the OpenAI format
    # args.type should be the reason type, e.g. L1, L2, L3
    # Using flatMap since convert_to_openai_format now returns multiple data points per row
    # Wrap each message list in a tuple so it can be converted to DataFrame properly
    df = df.rdd.flatMap(lambda x: [(messages,) for messages in convert_to_openai_format(x, args.type)]).toDF(["messages"])
    # df.show(10)
    for row in df.head(n=5):
        print("--" * 50)
        print(json.dumps(row, indent=4))
        print("--" * 50)
    spark.stop()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default="L2", choices=["L1", "L2", "L3"])
    args = parser.parse_args()
    main(args)

