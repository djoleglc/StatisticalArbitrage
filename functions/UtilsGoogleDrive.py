from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import shutil


def UploadFile(file, folder_id):
    gauth = GoogleAuth()
    drive = GoogleDrive(gauth)
    name_file = file.split("\\")[-1].split("/")[-1]
    gfile = drive.CreateFile({"parents": [{"id": folder_id}]})
    # Read file and set it as the content of this instance.
    gfile["title"] = name_file
    gfile.SetContentFile(file)
    gfile.Upload()  # Upload the file


def UploadFileListData(files, folder_id):
    gauth = GoogleAuth()
    drive = GoogleDrive(gauth)
    for file in files:
        name_file = file.split("\\")[-1].split("/")[-1]
        gfile = drive.CreateFile({"parents": [{"id": folder_id}]})
        # Read file and set it as the content of this instance.
        gfile["title"] = name_file
        gfile.SetContentFile(file)
        gfile.Upload()  # Upload the file


def CreateFolder(folder_id, folderName, return_id=False):

    gauth = GoogleAuth()
    drive = GoogleDrive(gauth)
    file_list = FilesAvailableDrive(folder_id)
    id_ = [j["id"] for j in file_list if j["title"] == folderName]
    if len(id_) > 0:
        return id_[0]
    file_metadata = {
        "title": folderName,
        "parents": [{"id": folder_id}],  # parent folder
        "mimeType": "application/vnd.google-apps.folder",
    }

    folder = drive.CreateFile(file_metadata)
    folder.Upload()
    if return_id:
        file_list = FilesAvailableDrive(folder_id)
        id_ = [j["id"] for j in file_list if j["title"] == folderName]
        if len(id_) > 0:
            return id_[0]


def FilesAvailableDrive(folder_id):
    gauth = GoogleAuth()
    drive = GoogleDrive(gauth)
    file_list = drive.ListFile(
        {"q": "'{}' in parents and trashed=false".format(folder_id)}
    ).GetList()
    return file_list


def RetrieveFile(file_list, folder_id):
    gauth = GoogleAuth()
    drive = GoogleDrive(gauth)

    for i, file in enumerate(sorted(file_list, key=lambda x: x["title"]), start=1):
        print(
            "Downloading {} file from GDrive ({}/{})".format(
                file["title"], i, len(file_list)
            )
        )
        file.GetContentFile(file["title"])


def CreatePairFolder(folder_id, pair):
    asset_1 = pair[0]
    asset_2 = pair[1]
    title = f"{asset_1}_{asset_2}"
    file_list = FilesAvailableDrive(folder_id)
    id_ = [j["id"] for j in file_list if j["title"] == title]
    if len(id_) > 0:
        return title, id_[0]

    CreateFolder(folder_id, title)
    files = FilesAvailableDrive(folder_id)
    id_ = [j["id"] for j in files if j["title"] == title][0]
    return title, id_


def RetrieveFolder(files, name_folder, output_folder="H:"):
    current_dir = os.getcwd()
    id_ = [j["id"] for j in files if j["title"] == name_folder][0]
    ls = FilesAvailableDrive(id_)
    RetrieveFile(ls, id_)

    title_ = [j["title"] for j in ls]

    path = os.path.join(current_dir, name_folder)
    all_file = os.listdir(current_dir)
    os.makedirs(path, exist_ok=True)

    for title in title_:
        name_file = [j for j in all_file if len(j.split(title)) > 1][0]
        path_to_save = f"{name_folder}/"
        shutil.move(name_file, path_to_save)

    if output_folder is not None:
        shutil.move(name_folder, output_folder)


def SaveUploadResultStrategy(
    result, asset_name_1, asset_name_2, idx, drive=True, output_folder="H:"
):
    folder_name = f"{asset_name_1}_{asset_name_2}"
    path_result = f"{folder_name}.joblib"
    joblib.dump(result, path_result)
    if output_folder is not None:
        shutil.move(path_result, f"{output_folder}")
        path_result = f"{output_folder}/{path_result}"

    if drive:
        files = FilesAvailableDrive(idx)
        id_ = [j["id"] for j in files if j["title"] == folder_name][0]
        UploadFile(file=path_result, folder_id=id_)
