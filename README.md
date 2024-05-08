## Show and Go

Step 1:
  * Install the python 3.10.11 version in your laptop
  * from the official python website.


Step 2:
  * Create a virtual environment in your system
  * Open command prompt
  * pip3 install virtualenv - installing virtual env
  * Go to any folder and create virtaulenv
  * virtualenv "envname" || e.g, virtualenv myenv - cerating virtualenv
  * In that folder type, "envname"\Scripts\activate.bat || e.g, myenv\Scripts\activate.bat


Step 3:
   * Copy the dlib file int the folder and Paste it in the user/Download folder in the Local disk(which is C drive)
   * Open requirements.txt and change the directory location
   * "dlib @ file:///C:/Users/kakar/Downloads/dlib-19.22.99-cp310-cp310-win_amd64.whl" with your file location.
   * Now this in command prompt, pip install -r requirements.txt
   * the above code will install all the dependencies in virtualenv


Step 4:
   * Now all the depencies are installed
   * Finally, run python website.py runserver - to run thr django project
   * To see this project in your mobile phone, run the command as python website.py runserver youripaddress:8000 (Connect with same wifi)
