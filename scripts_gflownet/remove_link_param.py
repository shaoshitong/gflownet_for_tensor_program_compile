import os

def find_empty_files(dir_path):
    # 检查路径是否存在
    if not os.path.exists(dir_path):
        print("指定的路径不存在")
        return []

    # 初始化空文件列表
    empty_files = []

    # 遍历文件夹
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)

            # 检查文件大小
            if os.path.getsize(file_path) == 0:
                empty_files.append(file_path)

    return empty_files

def main():
    dir_path = input("请输入需要检查的文件夹路径：")
    empty_files = find_empty_files(dir_path)

    if not empty_files:
        print("没有找到空文件")
    else:
        print("找到以下空文件：")
        for file in empty_files:
            print(file)
            # os.system(f"rm -rf {file}")

if __name__ == "__main__":
    main()
