import datasets

training_file = "WASSA_train.csv"
test_file = "WASSA_dev.csv"


def main():
    data_files = {"train": training_file,
                  "test": test_file}

    data = datasets.load_dataset("csv", data_files=data_files,
                                 delimiter=",",
                                 encoding='utf-8')

    print(data)


if __name__ == '__main__':
    main()
