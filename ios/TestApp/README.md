## TestApp

The TestApp is being used for different purposes as described below

### Cocoapods

To quickly test our framework in Cocoapods, simply run 

```ruby
pod install
```

This will pull the latest version of `LibTorch` from Cocoapods.

### Circle CI and Fastlane

The TestApp is currently being used as a dummy app by Circle CI for nightly jobs. The challenge comes when testing the arm64 build as we don't have a way to code-sign our TestApp. This is where Fastlane came to rescue. [Fastlane](https://fastlane.tools/) is a trendy automation tool for building and managing iOS applications. It also works seamlessly with Circle CI. We are going to leverage the `import_certificate` action, which can install developer certificates on CI machines. See `Fastfile` for more details.

### Benchmark

The benchmark folder contains two scripts that help you setup the benchmark project. The `setup.rb` does the heavy-lifting jobs of setting up the XCode project, whereas the `trace_model.py` is a Python script that you can tweak to generate your model for benchmarking. Simply follow the steps below to setup the project

1. In the PyTorch root directory, run `BUILD_PYTORCH_MOBILE=1 IOS_ARCH=arm64 ./scripts/build_ios.sh` to generate the custom build from **Master** branch
2. Navigate to the `benchmark` folder, run `python trace_model.py` to get your model generated.
3. In the same directory, run `ruby setup.rb` to setup the XCode project.
4. Open the `TestApp.xcodeproj`, you're ready to go.

The benchmark code is written in C++, see `benchmark.mm` for more details.

### `bootstrap.sh`

For those who want to do perf testing but don't want touch XCode, `bootstrap.sh` is the right tool for you. It'll automatically build and install the app on your device. That being said, it does require you to have 

1. A valid iOS dev certificate installed on your local machine. 
2. A valid provisioning profile for code signing
3. A valid team identifier

To run the script, simply type the command below and make sure your phone is unlocked and connected via USB. 

```shell
./bootstrap -t ${TEAM_ID} -p ${PROVISIONING_PROFILE}
```

The benchmark log will be displayed on the screen.

> Note This requires ios-deploy to be installed. Please have a look at [ios-deploy](https://github.com/ios-control/ios-deploy). To quickly install it, use `npm -g i ios-deploy`



