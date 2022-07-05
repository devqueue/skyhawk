# SKYHAWK
Skyhawk is a CLI application that uses face recognition to add attendance of registered users to a csv file and send it to a database

### Motivation
My mum being a teacher at my school I noticed she had to wait in a queue in the morning to use her fingerprint to mark attendance. Which made me think the school must appoint a clerk who would look at people entering the building and mark their attendance and so skyhawk was born to replace the clerk I envisioned. Well AI is definitely going to take away jobs

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://github.com/devqueue/Skyhawk-cli)

### Installing the App:

#### 1. Using the repo:
   Follow the given steps to test the code:
    1. `git clone https://github.com/devqueue/Skyhawk-cli.git`
    2. `cd Skyhawk-cli`
    3. `pip install -r requirements.txt`
    4. `pip install -e . `

### Testing the code:
You can initialize and register users and capture their faces.
Make sure to  hold the camera in front of your face in a good lighting condition.
1. `skyhawk init`
2. Add one clear image of each person with their identifier in `facedata`
3. `skyhawk train`
Now run the application.
4. `skyhawk run`
Now place a camera in at the entrance of the office or classroom
and let the registered folks walk past it. You can now view their attendance using the following command.
5. `skyhawk view attandance`


### Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

### License
[MLP 2.0](https://www.mozilla.org/en-US/MPL/2.0/)
