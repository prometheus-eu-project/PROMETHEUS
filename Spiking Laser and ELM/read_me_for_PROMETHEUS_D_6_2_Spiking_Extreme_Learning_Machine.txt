The **"PROMETHEUS_D_6_2_Spiking_Extreme_Learning_Machine_Simulation.m"** script simulates the behavior of a spiking extreme learning machine (ELM) based on a two-section gain saturable absorber quantum well laser. Specifically, it computes the output of an excitable two-section laser that is electrically injected with incoming image data. The operation of the ELM is divided into three key stages:  

1. **Preprocessing:**  
   In this stage, incoming images are organized into kernels. The elements within each kernel are flattened into a one-dimensional vector and multiplied by a randomly generated matrix, referred to as the "mask." The mask has dimensions of **(K² × L)**, where **K²** represents the number of elements per kernel, and **L** is an integer. Then, the mask and the flattened kernel are multiplied. As a result, each kernel is transformed into **L** values, which are random linear combinations of the elements of the kernel. This procedure is analogous to the masking process used in classical time-delay reservoir computing. Subsequently, the generated values are scanned in multiple directions to form the input for the next layer. This step is imperative as it enables the laser to preserve the spatial coherence of the pixels through its internal dynamics while simultaneously reducing the size of the input images (**lines 39–83**).  

2. **Main Processing:**  
   This section simulates the response of the two-section gain saturable absorber quantum well laser when injected with the preprocessed image data from the previous stage (**lines 86–111**).  

3. **Output Generation:**  
   The spiking output of the laser is converted into a binary representation. Specifically, the laser output is divided into multiple equal time slots, each characterized by a numerical value. If a spike occurs within a given time slot, its corresponding value is set to **1**; otherwise, it is set to **0**.  

Finally, the binary output is classified using a fully connected layer, which is implemented in Python using the **PyTorch** framework.