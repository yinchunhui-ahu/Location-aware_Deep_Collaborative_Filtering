****************************************************************************
* README file for 339 * 5825 Web service QoS dataset
****************************************************************************

This dataset describes real-world QoS evaluation results from 339 users on 
5,825 Web services. It is available for downloading at:
http://wsdream.github.io/dataset/wsdream_dataset1.html

****************************************************************************
List of contents of the dataset
****************************************************************************

rt(tp)_origin.txt:
- Instances of response-time(or throughput) converted from the original user-service matrix.
- Train|Test instances with location
- Format: User ID|Service ID|Response-time

rt(tp)_train(test)_density.txt:
- Instances of response-time(or throughput) for training(or testing) at corresponding density.
- Format: User ID|Service ID|Response-time|User Country|User AS|Service Country|Service AS

readme.txt:
- Descriptions of the dataset provided.
