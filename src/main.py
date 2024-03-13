from one_dimensional.OneDimensionalEnvironment import OneDimensionalEnvironment
import yaml


def main():

    # # Define your tuple
    # my_tuple = ((0, (0, 1)), (1, (1, 1)))
    #
    # # Define the file path
    # file_path = 'tuple.yaml'
    #
    # # Write the tuple to a YAML file
    # with open(file_path, 'w') as file:
    #     yaml.dump({'node_name': {
    #         'value': 1,
    #         'parents': ['node_name'],
    #         'CPT': my_tuple
    #     }}, file)
    env = OneDimensionalEnvironment()


if __name__ == "__main__":
    main()
