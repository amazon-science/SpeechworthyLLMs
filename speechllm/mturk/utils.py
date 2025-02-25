import boto3 

def associate_qualification_with_worker(worker_id, integer_value=100):
    client = boto3.session.Session(profile_name='mturk').client('mturk')

    response = client.associate_qualification_with_worker(
        QualificationTypeId='3B5O8SC7Q7AHWK1W0BWSJU39IPFSL0',
        WorkerId=worker_id,
        IntegerValue=integer_value,
        SendNotification=False
    )

    # Optionally, check the response for any issues
    if 'ResponseMetadata' in response and response['ResponseMetadata']['HTTPStatusCode'] != 200:
        print(f"Failed to associate qualification with worker {worker_id}")


def get_all_workers_with_qualification(qualification_type_id):

    # Initialize a session using Amazon MTurk
    client = boto3.session.Session(profile_name='mturk').client('mturk')

    # Paginator to handle potential pagination of results
    paginator = client.get_paginator('list_workers_with_qualification_type')

    # List all workers with status Active, Inactive, or Revoked
    # Adjust the 'Statuses' as necessary.
    page_iterator = paginator.paginate(
        QualificationTypeId=qualification_type_id,
    )

    workers = [] 
    # Loop through each page of results
    for page in page_iterator:
        # Loop through each qualification in the page
        for qualification in page['Qualifications']:
            workers.append(qualification)

    print(f"Found {len(workers)} workers with qualification {qualification_type_id}")

    return workers 

def remove_workers_from_qualification(qualification_type_id):
    # Initialize a session using Amazon MTurk
    client = boto3.session.Session(profile_name='mturk').client('mturk')

    # Get the list of workers associated with the qualification
    workers = get_all_workers_with_qualification(qualification_type_id)

    # Loop through each worker and disassociate the qualification
    for worker in workers: 
        worker_id = worker['WorkerId']
        client.disassociate_qualification_from_worker(
            WorkerId=worker_id,
            QualificationTypeId=qualification_type_id,
            Reason='Removing all workers from this qualification.'  # Optional reason
        )
        print(f"Removed qualification from worker {worker_id}")


def list_all_qualification_types():
    # Initialize a session using Amazon MTurk
    client = boto3.session.Session(profile_name='mturk').client('mturk')

    # Paginator to handle potential pagination of results
    paginator = client.get_paginator('list_qualification_types')

    # List all qualification types with status Active, Inactive, and Updating
    # Adjust the 'MustBeRequestable' and 'MustBeOwnedByCaller' as necessary.
    page_iterator = paginator.paginate(
        Query="vllm", 
        MustBeRequestable=False,
        MustBeOwnedByCaller=True
    )

    for page in page_iterator:
        for qualification_type in page['QualificationTypes']:
            print("Name:", qualification_type['Name'])
            print("ID:", qualification_type['QualificationTypeId'])
            print("Description:", qualification_type['Description'])
            print("Status:", qualification_type['QualificationTypeStatus'])
            print("-----")
