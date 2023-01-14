from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive


def UploadFile(file, folder_id):
    gauth = GoogleAuth()          
    drive = GoogleDrive(gauth)
    name_file = file.split("/")[-1].split(".csv")[0]
    gfile = drive.CreateFile({'parents': [{'id': folder_id}]})
    # Read file and set it as the content of this instance.
    gfile["title"] = name_file
    gfile.SetContentFile(file)
    gfile.Upload() # Upload the file
        

def UploadFileListData(files, folder_id):
    gauth = GoogleAuth()          
    drive = GoogleDrive(gauth)
    for file in files:
        name_file = file.split("\\")[-1]
        gfile = drive.CreateFile({'parents': [{'id': folder_id}]})
        # Read file and set it as the content of this instance.
        gfile["title"] = name_file
        gfile.SetContentFile(file)
        gfile.Upload() # Upload the file
        
        
def CreateFolder(folder_id, folderName, return_id = False):
            
        gauth = GoogleAuth()          
        drive = GoogleDrive(gauth)
        file_list = FilesAvailableDrive(folder_id)
        id_ = [ j["id"] for j in file_list if j["title"] == folderName]
        if len(id_)>0:
            return id_[0]
        file_metadata = {
            'title': folderName,
            'parents': [{'id': folder_id}], #parent folder
            'mimeType': 'application/vnd.google-apps.folder'
        }

        folder = drive.CreateFile(file_metadata)
        folder.Upload()
        if return_id:
            file_list = FilesAvailableDrive(folder_id)
            id_ = [ j["id"] for j in file_list if j["title"] == folderName]
            if len(id_)>0:
                return id_[0]
            
        
        
def FilesAvailableDrive(folder_id):
    gauth = GoogleAuth()          
    drive = GoogleDrive(gauth)
    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(folder_id)}).GetList()
    return file_list
    

def RetrieveFile(file_list, folder_id):
    gauth = GoogleAuth()          
    drive = GoogleDrive(gauth)

    for i, file in enumerate(sorted(file_list, key = lambda x: x['title']), start=1):
        print('Downloading {} file from GDrive ({}/{})'.format(file['title'], i, len(file_list)))
        file.GetContentFile(file['title'])
    
    
def CreatePairFolder(folder_id, pair):
    asset_1 = pair[0]
    asset_2 = pair[1]
    title =  f"{asset_1}_{asset_2}"
    file_list = FilesAvailableDrive(folder_id)
    id_ = [ j["id"] for j in file_list if j["title"] == title]
    if len(id_)>0:
        return title,id_[0]
    
    CreateFolder(folder_id,title)
    files = FilesAvailableDrive(folder_id)
    id_ = [ j["id"] for j in files if j["title"] == title][0]
    return title,id_
    

    
    