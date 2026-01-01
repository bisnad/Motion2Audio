# Motion2Audio

Motion2Audio is a research project that aims to develop AI-based tools for translating dance movements into music. These tools are designed to allow dancers to freely improvise to existing music and then use recordings of these improvisations as the basis for interactively controlling the creation of music through their body movements.

### Artistic Principle

At the core of this project lies the idea that performers in contemporary dance have developed highly refined techniques and strategies for using music as a creative resource in the generation of movement. Motion2Audio seeks to adopt these techniques as the foundation for developing digital musical instruments whose interaction and sound generation are guided solely by the dancers’ idiosyncratic decisions made while improvising to music.

### Technical Principe

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





