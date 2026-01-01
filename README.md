# Motion2Audio

Motion2Audio is a research project that aims to develop AI-based tools for translating dance movements into music. The tools are meant to allow dancers to freely improvise to existing music and then use recordings of these improvisations as basis for interactively controlling the creation of music by means of their body movements. 

### Artistic Principle

At the core of this project lies the idea that performers in contemporary dance have developed  idiosyncratic techniques and strategies to use music as a resource for ideating movement. Motion2Audio tries to adopt the dancer's improvisation techniques as basis to create digital musical instruments whose interaction and music generation principles are solely informed by creative decision as to how to relate his or her movement to music. 

### Technical Principe

Motion2Audio develops machine learning models that analyse a dancer's movements and translates these movements into music through neural sound synthesis. These models are trained on music and movement recordings of dancers who have been improvising to music. Through training, the models learn the correlations between movement and music. Once trained, the models employ these correlations to generate new music from movement alone. 

### Repository

This repository is divided into the following sections. 

- The [MotionCapture](https://github.com/bisnad/Motion2Audio/tree/main/MotionCapture) section contains tools for converting proprietary message protocols of motion capture systems to the OSC (open sound control) and for playing motion capture recordings.
- The [Transformer](https://github.com/bisnad/Motion2Audio/tree/main/Transformer) section contains tools for training and using Transformer models that translate motion to audio.
-  The [VAE](https://github.com/bisnad/Motion2Audio/tree/main/VAE) section contains for training and using Variational Autoencoders for compressing and reconstructing audio.

### Partners

Currently, the projects runs as a collaboration between two researchers and three professional dancers.

**Researchers**

- Daniel Bisig, Institute for Computer Music and Sound Technology, Zurich University of the Arts (https://www.zhdk.ch/en/research/icst)
- Alexander Okupnik, Artificial Intelligence and Data Science, University Liechtenstein (https://www.uni.li/en/university/organisation/liechtenstein-business-school/artificial-intelligence-and-data-science)

**Dancers**

- Diane Gemsch (https://www.dianegemsch.ch/)
- Eleni Mylona (https://www.mylonaeleni.com/)
- Tim Winkler





