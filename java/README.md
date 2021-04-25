## Getting Started

Welcome to the VS Code Java world. Here is a guideline to help you get started to write Java code in Visual Studio Code.

## Folder Structure

The workspace contains two folders by default, where:

- `src`: the folder to maintain sources
- `lib`: the folder to maintain dependencies

## Dependency Management

The `JAVA DEPENDENCIES` view allows you to manage your dependencies. More details can be found [here](https://github.com/microsoft/vscode-java-pack/blob/master/release-notes/v0.9.0.md#work-with-jar-files-directly).

## Execution

Windows Power Shell:
```
# Create a build directory parallel to the src directory
mkdir build
# Build the code into the build directory
javac -d build ./src/*.java
# Execute the code
java -cp build App

# Single line
javac -d build src/*.java; java -cp build App
```