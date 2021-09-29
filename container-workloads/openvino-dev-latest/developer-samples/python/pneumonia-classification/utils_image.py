import os
import matplotlib
import matplotlib.pyplot as plt

def show_results(result_dir, output_file):
    result_file=result_dir+f"/{output_file}"
    file_ready= os.path.isfile(result_file)
    
    if file_ready:
        count=0
        with open(result_file) as f:
            for ind, line in enumerate(f):
                if line=="\n":
                    break
                print(line)
                image_file=result_dir+'/result_'+str(ind)+'.png'
                im=plt.imread(image_file)
                plt.figure(figsize = (20,20))
                plt.box(False)
                plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
                plt.imshow(im)
                plt.show()
    else: 
        print("The results are not ready yet, please retry")
