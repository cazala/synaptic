Test using gulp, from root directory:

`gulp test`

To test the web version, start a web server at the root dir of this repo, then use your OS browser. 

[Python](http://xkcd.com/353/) to the rescue!  From command line run this command:

```bash
$ python -m SimpleHTTPServer 8080
#go to http://localhost:8080/test/test_browser.html
```

Or you can use node.js instead:

```bash
$ npm install -g http-server
```
Then run:

```bash
$ http-server
#go to http://localhost:8080/test/test_browser.html
```


