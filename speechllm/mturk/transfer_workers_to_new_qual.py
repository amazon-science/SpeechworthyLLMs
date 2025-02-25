from speechllm.mturk.utils import get_all_workers_with_qualification, remove_workers_from_qualification, list_all_qualification_types, associate_qualification_with_worker

from argparse import ArgumentParser

def main():

    parser = ArgumentParser()
    parser.add_argument("--original_qual_id", "-o", help="The qualification ID to remove from workers", default="3JB3E7VN4TNPIQKVJXP8JG73VAM9PX")
    parser.add_argument("--original_qual_threshold", "-t", help="The threshold for the original qualification", default=66)
    parser.add_argument("--new_qual_id", "-n", help="The qualification ID to add to workers", default="3B5O8SC7Q7AHWK1W0BWSJU39IPFSL0")
    parser.add_argument("--new_qual_value", "-v", help="The value of the new qualification", default=100)
    parser.add_argument("--reset", "-r", help="Remove all workers in the new qualification", action="store_true")
    parser.add_argument("--check", "-c", help="Check the number of workers in the new qualification", action="store_true")

    args = parser.parse_args()

    list_all_qualification_types()

    if args.reset:
        remove_workers_from_qualification(args.new_qual_id)
        return 
    
    if args.check: 
        get_all_workers_with_qualification(args.new_qual_id)
        return
    
    # get a list of workers with qualification value that exceed the threshold
    all_original_workers = get_all_workers_with_qualification(args.original_qual_id)

    
    for worker in all_original_workers:
        worker_id   = worker['WorkerId']
        if worker['IntegerValue'] >= args.original_qual_threshold:
            associate_qualification_with_worker(worker_id, integer_value=args.new_qual_value)
            print(f"Added qualification to worker {worker_id}")
        else:
            print(f"Worker {worker_id} did not meet the threshold: {worker['IntegerValue']}")


if __name__ == "__main__":
    main()
