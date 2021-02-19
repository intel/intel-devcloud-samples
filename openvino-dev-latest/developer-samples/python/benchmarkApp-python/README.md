# Intel DevCloud Benchmark Notebook
This repository contains Jupyter notebook files for benchmarking models in Intel DevCloud.
## Instructions
### 1. Upload models to AWS S3 bucket
Upload your models converted to OpenVINO optimized model files (IR Files) in zip format.
* **File Format**: [Model_Name].zip  
* **Example**: MobileNetV2-keras.zip   
* **Folder structure after unzip**: [Model_Name]/IR_models/[FP16 or FP32]/[IR FILE NAME].xml  

>**Note**: S3 bucket should not contain any other files or folder. 
### 2. Create new IAM user and generate AWS credentials
1. Sign in to [IAM Console](https://console.aws.amazon.com/iam/home#/home)
2. Create a new IAM User with programmatic access 
3. Attach 'AmazonS3FullAccess' & 'AmazonSESFullAccess' policy to IAM User
4. Create Access Key - Please download 'Download.csv' file for future reference. AWS Access keys are used to configure AWS SDK.  
>For detailed instructions, Please follow this [link](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html#id_users_create_console)  

### 3. Verify Email Address 
The Benchmark notebook uses AWS SES service for sending email notifications on completing the model benchmarking.
Before you can send email from your email address through Amazon SES, you need to show Amazon SES that you own the email address by verifying it.  
>For instructions, see [Verifying email addresses in Amazon SES](https://docs.aws.amazon.com/ses/latest/DeveloperGuide/verify-email-addresses-procedure.html)

### 4. Running OpenVINO Benchmark App in Intel Dev Cloud
1. Download BenchmarkApp.ipynb, device_link.json and requirements.txt from this repository.  
2. Sign In to your [Intel DevCloud instance](https://devcloud.intel.com/edge/) .
>For new Devcloud users, refer this [link](https://devcloud.intel.com/edge/get_started/guide/) to register & sign in.  
3. Click on **Advanced** --> [**Connect and Create**](https://devcloud.intel.com/edge/advanced/connect_and_create/) and click on **My Files**
4. Go inside **My-Notebooks** folder and create a new folder
5. Upload downloaded files(BenchmarkApp.ipynb, device_link.json & requirements.txt) to newly created folder in Devcloud notebook instance.
6. Run BenchmarkApp.ipynb and follow the instruction in Jupyter notebook.
>**Tip**  
>You can provide required details in notebook and click on **Click --> Run All** in your Jupyter notebook to run all the cells.  You will recieve an email notification once benchmarking is completed
