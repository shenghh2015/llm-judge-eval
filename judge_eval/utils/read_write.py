import typing
import csv


def csv_file_read(file_name, max_num: int = -1) -> typing.Iterator:
  assert file_name.endswith(".csv")
  data_num = 0
  with open(file_name, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      data_num += 1
      if max_num >= 0 and data_num > max_num:
        break

      yield row


def csv_file_write(data: typing.Iterator, field_names: list, file_name,
                   **kwargs):
  assert file_name.endswith(".csv")
  with open(file_name, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names)
    writer.writeheader()
    for d in data:
      writer.writerow(d)


def jsonl_file_read(file_name, max_num: int = -1) -> typing.Iterator:
  assert file_name.endswith(".jsonl")
  data_num = 0
  with open(file_name, encoding="utf-8") as fin:
    for idx, ln in enumerate(fin):
      if max_num >= 0 and idx + 1 > max_num:
        break

      try:
        obj = eval(ln)
        yield obj
        data_num += 1

      except Exception as err:
        print("read errors")


def jsonl_file_write(data: typing.Iterator, file_name: str, **kwargs) -> None:
  assert file_name.endswith(".jsonl")
  if isinstance(data, dict):
    data = [data]
  with open(file_name, "w") as fou:
    num = 0
    for obj in data:
      num += 1
      print(obj, file=fou)
