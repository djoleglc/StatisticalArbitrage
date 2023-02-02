from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os
import shutil
import joblib



def UploadFile(file, folder_id):
    """
    Upload a file to a specified Google Drive folder.

    Inputs:
        - file: str
                The file path to upload.
        - folder_id: str
                The ID of the Google Drive folder to upload the file to.

    Output: None
    """
    gauth = GoogleAuth()
    drive = GoogleDrive(gauth)
    name_file = file.split("\\")[-1].split("/")[-1]
    gfile = drive.CreateFile({"parents": [{"id": folder_id}]})
    # Read file and set it as the content of this instance.
    gfile["title"] = name_file
    gfile.SetContentFile(file)
    gfile.Upload()

    
    
    
def UploadFileListData(files, folder_id):
    """
    Function to upload a list of files to a folder in Google Drive.
    
    Input:
      - files: list
            List of files to be uploaded.
      - folder_id: str
            ID of the Google Drive folder where the files will be uploaded to.
            
    Output:
      - None
    """
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
    """
    Function to create a folder in Google Drive.
    
    Input:
      - folder_id: str
            ID of the parent folder in Google Drive where the new folder will be created in.
      - folderName: str
            Name of the folder to be created.
      - return_id: bool
            Flag indicating whether the ID of the created folder should be returned or not.
            
    Output:
      - folder_id: str (if return_id is True)
            ID of the created folder in Google Drive.
    """
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
    """
    Function to retrieve a list of files in a Google Drive folder.
    
    Input:
      - folder_id: str
            ID of the Google Drive folder to retrieve the list of files from.
            
    Output:
      - file_list: list
            List of files in the Google Drive folder.
    """
    gauth = GoogleAuth()
    drive = GoogleDrive(gauth)
    file_list = drive.ListFile(
        {"q": "'{}' in parents and trashed=false".format(folder_id)}
    ).GetList()
    return file_list





def RetrieveFile(file_list, folder_id):
    """
    Function to retrieve files from a Google Drive folder.
    
    Input:
      - file_list: list
            List of files in the Google Drive folder.
      - folder_id: str
            ID of the Google Drive folder to retrieve the files from.
            
    Output:
      None
    """
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
    """
    Function to create a folder in Google Drive with the title derived from a pair of assets.
    
    Input:
      - folder_id: str
            ID of the parent Google Drive folder.
      - pair: list
            Pair of assets to derive the title for the folder.
            
    Output:
      - title: str
            Title of the folder.
      - id_: str
            ID of the folder.
    """
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
    """
    Function to retrieve a Google Drive folder and move it to a local directory.
    
    Input:
      - files: list
            List of files in Google Drive.
      - name_folder: str
            Name of the Google Drive folder to be retrieved.
      - output_folder: str (optional)
            Path to the local directory where the retrieved folder should be moved to. Default is "H:".
            
    Output:
      None
    """
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
    """
    Function to save the results of a calculation and upload the saved file to Google Drive.
    
    Input:
      - result: object
            The result of the calculation to be saved.
      - asset_name_1: str
            First name of the asset.
      - asset_name_2: str
            Second name of the asset.
      - idx: str
            ID of the Google Drive folder where the saved file should be uploaded to.
      - drive: bool (optional)
            Boolean value indicating whether to upload the saved file to Google Drive or not. Default is True.
      - output_folder: str (optional)
            Path to the local directory where the saved file should be stored. Default is "H:".
            
    Output:
      None
    """
    folder_name = f"{asset_name_1}_{asset_name_2}"
    path_result = os.path.join(output_folder, f"{folder_name}/{folder_name}.joblib")
    joblib.dump(result, path_result)

    if drive:
        files = FilesAvailableDrive(idx)
        id_ = [j["id"] for j in files if j["title"] == folder_name][0]
        UploadFile(file=path_result, folder_id=id_)
