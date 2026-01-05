# Motion2Audio

Motion2Audio is a research project that aims to develop AI-based tools for translating dance movements into music. These tools are designed to allow dancers to freely improvise to existing music and then use recordings of these improvisations as the basis for interactively controlling the creation of music through their body movements.

https://github.com/user-attachments/assets/904f2080-76e2-4f6d-9165-09b0d59ebc91

figure 1: video and motion capture recording of dancer Diane Gemsch improvising to the audio recording of her father's voice. 

https://github.com/user-attachments/assets/941ba15c-837c-4dd3-b066-4181bc02b3c4

figure 2: video and motion capture recording of dancer Tim Winkler improvising to a music that has been composed for choreographer Saju Hari.

### Artistic Principle

At the core of this project lies the idea that performers in contemporary dance have developed highly refined techniques and strategies for using music as a creative resource in the generation of movement. Motion2Audio seeks to adopt these techniques as the foundation for developing digital musical instruments whose interaction and sound generation are guided solely by the dancers’ idiosyncratic decisions made while improvising to music.

### Technical Principle

Motion2Audio develops machine learning models that analyze a dancer's movements and translate these movements into music through neural sound synthesis. These models are trained on recordings of dancers improvising to music. Through training, the models learn the correlations between movement and sound. Once trained, they can generate new music from movement alone.

### Repository

This repository is divided into the following sections. 

- The [MotionCapture](https://github.com/bisnad/Motion2Audio/tree/main/MotionCapture) section contains tools for converting proprietary motion capture message protocols to OSC (Open Sound Control) format and for playing back motion capture recordings.
- The [Transformer](https://github.com/bisnad/Motion2Audio/tree/main/Transformer) section contains tools for training and using Transformer models that translate motion into audio.
-  The [VAE](https://github.com/bisnad/Motion2Audio/tree/main/VAE) section contains tools for training and using Variational Autoencoders to compress and reconstruct audio.

### Partners

Currently, the project runs as a collaboration between two researchers and three professional dancers.

**Researchers**

- Daniel Bisig, Institute for Computer Music and Sound Technology, Zurich University of the Arts (https://www.zhdk.ch/en/research/icst)
- Alexander Okupnik, Artificial Intelligence and Data Science, University Liechtenstein (https://www.uni.li/en/university/organisation/liechtenstein-business-school/artificial-intelligence-and-data-science)

**Dancers**

- Diane Gemsch (https://www.dianegemsch.ch/)
- Eleni Mylona (https://www.mylonaeleni.com/)
- Tim Winkler

### Example Results

https://github.com/user-attachments/assets/31a4c7a4-bed3-44f2-87df-d9498e2440ec

figure 3: Results of Motion2Audio trained on movements and audio recordings of Diane Gemsch

https://github.com/user-attachments/assets/d82dc01f-1c64-4ab2-95b2-bb7d505a574f

figure 4: Results of Motion2Audio trained on movements and audio recordings of Tim Winkler




