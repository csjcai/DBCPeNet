
### Bright and Dark Channel extractor

#### modify the caffe.prototxt
```
optional WbcPriorParameter wbcprior_param = 176;
```

```
message WbcPriorParameter {
  enum PriorMethod {
    DARK = 1;
    WHITE = 2;
  }
  optional PriorMethod typeprior = 1 [default = DARK];
  optional uint32 kernel_size = 2 [default = 31];
}
```

#### Parameter
1. DARK: dark channel extractor
2. WHITE: bright channel extractor
3. kernel_size: window of extractor



#### Usage
```
layer {
  name: "Whitechannel"
  type: "Wbcprior"
  bottom: "input"
  top: "output"
  wbcprior_param {
     typeprior: WHITE
     kernel_size: 31
  }
}
```
```
layer {
  name: "Darkchannel"
  type: "Wbcprior"
  bottom: "input"
  top: "output"
  wbcprior_param {
     typeprior: DARK
     kernel_size: 31
  }
}
```

