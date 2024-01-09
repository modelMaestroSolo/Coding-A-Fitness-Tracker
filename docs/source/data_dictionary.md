
# Data dictionary  

The dataset consists wristband accelerometer and gyroscope data obtained during free weights workout. It contains data of 5 participants performing various barbell exercises with a medium to heavy weight. 

frequencies at which the data was collected: 
accelerometer: 12.500HZ
gyroscope: 25.00HZ

## Accelerometer
Accelerometers measure acceleration, which is the rate of change of velocity with respect to time. They detect changes in linear motion and acceleration forces. Most accelerometers measure acceleration along three axes: X, Y, and Z. These axes represent three-dimensional space. Commonly used in devices such as smartphones, fitness trackers, and game controllers to detect changes in device orientation and movement.
Used in automotive applications for airbag deployment and stability control systems.
Provides information about the static and dynamic acceleration experienced by the device.
Output is typically in units of g-force (1g is equal to the acceleration due to gravity).

## Gyroscope
Gyroscopes measure angular velocity, which is the rate of rotation around a particular axis. They are sensitive to changes in orientation and rotation.Gyroscopes can measure rotational movement around three axes: X, Y, and Z. They also represent three-dimensional space. Used in applications where precise knowledge of angular velocity and changes in orientation is essential.
Commonly found in inertial navigation systems, drones, and image stabilization systems in cameras.Provides information about the rate of rotation rather than linear acceleration.
Output is typically in degrees per second (°/s).

| Variable Name | Type          | Description                         | Possible Values (if categorical) | Data Range (if numerical) | Missing Values | Data Quality Checks |
|---------------|---------------|-------------------------------------|-----------------------------------|--------------------------|-----------------|----------------------|
| epoch(ms)           | Int      | It measures time by the number of milliseconds that have elapsed since 00:00:00 UTC on 1 January 1970, the beginning of the Unix epoch.               | N/A                               | 0 to 100                 | 2%              | No negative values   |
| Gender        | Categorical   | Gender of the individual            | Male, Female, Other               | N/A                      | 1%              | Valid categories     |
| Income        | Numerical      | Monthly income in dollars           | N/A                               | $0 to $100,000           | 5%              | No negative values   |
| Education     | Categorical   | Highest education level attained    | High School, Bachelor's, Master's | N/A                      | 3%              | Valid categories     |
| Education     | Categorical   | Highest education level attained    | High School, Bachelor's, Master's | N/A                      | 3%              | Valid categories     |
| Education     | Categorical   | Highest education level attained    | High School, Bachelor's, Master's | N/A                      | 3%              | Valid categories     |