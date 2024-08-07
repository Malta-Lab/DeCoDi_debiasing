import pickle
import random
import os
import argparse

def gen_random_seeds(num_of_images, start, end, directory='pickle files'):
    """Generate a list of unique random seeds and save to a file."""
    
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_name = f"rnd_list_{start}_{end}_{num_of_images}.pkl"
    file_path = os.path.join(directory, file_name)
    
    try:
        rnd_list = random.sample(range(start, end), num_of_images)
        with open(file_path, 'wb') as f:
            pickle.dump(rnd_list, f)
        print(f"Random seeds successfully saved to {file_path}.")
    except ValueError as e:
        print(f"Error: {e}. Ensure the range size is at least as large as the number of images.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate random seeds and save to a pickle file.')
    parser.add_argument('start', type=int, help='The start of the range.')
    parser.add_argument('end', type=int, help='The end of the range.')
    parser.add_argument('num_of_images', type=int, help='The number of random seeds to generate.')
    args = parser.parse_args()

    gen_random_seeds(args.num_of_images, args.start, args.end)
