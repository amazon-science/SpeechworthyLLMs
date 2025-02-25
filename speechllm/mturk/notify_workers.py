from speechllm.mturk.utils import get_all_workers_with_qualification, remove_workers_from_qualification, list_all_qualification_types, associate_qualification_with_worker
from argparse import ArgumentParser
import boto3 

parser = ArgumentParser()
parser.add_argument("--qualification-type-id", "-q", help="Workers to notify", default="3JB3E7VN4TNPIQKVJXP8JG73VAM9PX")

args = parser.parse_args()

workers = get_all_workers_with_qualification(args.qualification_type_id)

print(f"Found {len(workers)} workers with qualification {args.qualification_type_id}")


def notify_worker(worker_id, subject, message): 

    client = boto3.session.Session(profile_name='mturk').client('mturk')

    client.notify_workers(
        Subject=subject,
        MessageText=message,
        WorkerIds=[
            worker_id,
        ]
    )


subject = ""
message = ""


for worker in workers: 
    worker_id = worker['WorkerId']
    notify_worker(worker_id, subject, message)