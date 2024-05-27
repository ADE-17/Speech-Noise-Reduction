# Speech Noise Reduction

This repository contains the code and documentation for the Speech Noise Reduction project, developed as part of the Kickelhack Hackathon held in Ilmenau, Germany by Fraunhofer IDMT. Our project was awarded first place in the competition.

## Table of Contents

# Speech Noise Reduction

This repository contains the code and documentation for the Speech Noise Reduction project, developed as part of the Kickelhack Hackathon held in Ilmenau, Germany by Fraunhofer IDMT. Our project was awarded first place in the competition.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Streamlit Application](#streamlit-application)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

The Speech Noise Reduction project aims to enhance the quality of speech recordings by reducing background noise using advanced signal processing techniques. This project was developed during the Kickelhack Hackathon, organized by Fraunhofer IDMT, to provide an effective solution for improving audio clarity in various applications, such as teleconferencing, hearing aids, and voice-controlled systems.

## Features

- **Noise Reduction**: Implements state-of-the-art algorithms to minimize background noise.
- **Real-Time Processing**: Capable of processing audio in real-time for live applications.
- **User-Friendly Interface**: Simple command-line interface for easy integration and use.
- **Cross-Platform Compatibility**: Works on Windows, macOS, and Linux.
- **Streamlit Application**: User interface for noise reduction, including options for audio and video file uploads.

## Installation

To get started with the Speech Noise Reduction project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/ADE-17/Speech-Noise-Reduction.git
    cd Speech-Noise-Reduction
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The Speech Noise Reduction tool can be used via the command line. Below are some common commands:

1. **Basic Noise Reduction**:
    ```bash
    python noise_reduction.py --input_file path/to/input.wav --output_file path/to/output.wav
    ```

2. **Real-Time Noise Reduction**:
    ```bash
    python real_time_noise_reduction.py
    ```

## Examples

Here are some examples to demonstrate the usage of the tool:

1. **Reducing Noise in a Pre-recorded File**:
    ```bash
    python noise_reduction.py --input_file samples/noisy_speech.wav --output_file results/clean_speech.wav
    ```

2. **Running Real-Time Noise Reduction**:
    ```bash
    python real_time_noise_reduction.py
    ```

## Streamlit Application

We have developed a Streamlit application for a more interactive and user-friendly experience. The application allows users to upload audio or video files, apply noise reduction, and download the processed files.

### Running the Streamlit App

To run the Streamlit application, use the following command:

```bash
streamlit run app.py
```
### Features of the Streamlit App

Audio Upload: Upload audio files for noise reduction processing.Video Upload: Upload video files; the application will extract the audio, apply noise reduction, and re-embed the clean audio into the video.Download Processed Files: Download the cleaned audio or video files directly from the interface.

## Contributing

We welcome contributions from the community. To contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

Please ensure that your code adheres to our coding standards and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

This project was developed during the Kickelhack Hackathon in Ilmenau, Germany, organized by Fraunhofer IDMT. We are proud to have been awarded first place for our efforts. We would like to thank all the participants and organizers for their support and contributions.
