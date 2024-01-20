# PyClassify
Python TensorFlow ResNet50 HTTP image classification server

This is a result of Google's TensorFlow team no longer supporting Go, a programming language made with their own (El Goog's) money.

Because I want to write my crawlers, which require image classification, in Go, but Python is the only Tensorflow supported language I really know, I have to do this somewhere, but my primary server for the site this is operating in service of _doesn't even have enough ram to load tensorflow_.

This is not built with any Python Web Framework, instead relying on the very basic http.server 
it expects a POST request to the endpoint `/classify` with a JSON-encoded JPEG image  

```json
{"image": "data:image/jpeg;base64,iVBORw0KGgoAAA...}
```

with or without the `data:image/jpeg;base64,`, that's just something my JS wanted to add.


it will respond with something like 
```
{
    "success": true,
    "upload_size": 1,
    "classifications": {
        "television": "0.85180295",
        "desktop_computer": "0.09160688",
        "monitor": "0.04095296",
        "screen": "0.006704196",
        "notebook": "0.0039336607",
        "home_theater": "0.002433881",
        "iPod": "0.0011253094",
        "entertainment_center": "0.00035410884",
        "web_site": "0.00026599315",
        "desk": "0.00025780668"
    }
}
```

always the top 10 classifications and all their weights or some kind of error.
