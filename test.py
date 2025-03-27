import shelve

file_path = 'data/datasets/init/processed/init'
with shelve.open(file_path) as file:
    print(list(file.keys()))