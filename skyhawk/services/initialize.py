import os
from skyhawk.misc.colors import COLORS,colorText


def run():
    current_dir = os.getcwd()
    faces_dir = "skyhawk/facedata"
    bin_dir = "skyhawk/bin"
    face_path = os.path.join(current_dir,faces_dir)
    bin_path = os.path.join(current_dir,bin_dir)
    os.makedirs(face_path, exist_ok=True)
    os.makedirs(bin_path, exist_ok=True)
    with open('skyhawk/bin/Attendance.csv', 'w') as file:
        HEADER = "prn, Date, Time, Day "
        file.write(HEADER)
    # with open('skyhawk/bin/trainer.py','w') as py:
    #   py.write(code)

    header = "[[green]]            Welcome to skyhawk! Auto-attandance cli tool      "
    footer = "[[white]]Data directories initialized sucessfully"
    Logo = '''[[cyan]]
                                                                                                        
                                   -://-`                                               
                                 /dNMMMNmo`                                             
                                +MMMy+sNMMy                                             
                                hMMm`  yMMN`                                            
                      .--.`     :NMMmhdMMNo      .-:-`                                  
                    :hNNNNd/     -sdMMMmy:     :hNNNNd+                                 
                   -NMd::yMM:     .sNMMh-     .NMd:-yMM/                                
                   -NMd::hMM:    :mMMMMMN+    .NMm::yMM/                                
            `.`     :hNNNMm/     :MMMMMMMo     :dMMNNd+     `..                         
          -ymdmh:    `.:mMNho.   `mMMMMMM-   `+ymMN+.`    .yddmh/                       
         `mMo-+MN.     yMMMMMy    sMMMMMd    +MMMMMd`     dMs-/NN-                      
          yMdohMd`     +NMMMMm    -MMMMM+    yMMMMMs`    `yMdoyMd`                      
          `:oysNmdh/    /NMMMM.    dMMMN.   `mMMMMs`   :hdmMyys/`                       
              .NMMMN/    /NMMM/    +MMMy    -MMMMo    -mMMMM/                           
              `/dMMMN/    :mMMy    .NMM:    +MMN+    -mMMMmo`                           
                `/dMMN:    -mMm`    hMm`    yMN/    .mMMm+.                             
                  `/hNm-    -mM.    /Ms    `mN/    .dMd+`                               
                     :hd-    .d/    `m-    .m:    `hd/`                                 
                      `:o`    ./     :     :-    `+d`                                    
                                                                                                     
    '''
    print('\n')
    print(colorText(header))
    print(colorText(Logo))
    print(colorText(footer))
    print('\n')
