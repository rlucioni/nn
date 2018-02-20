# nn

Neural network practice

## Quickstart

Install requirements:

```
$ make requirements
```

Initialize models and drop into Python's interactive mode:

```
$ make run
```

In interactive mode, restore variables (learned model state):

```py
>>> bad.restore()
>>> good.restore()
```

Test models:

```py
>>> bad.test()
>>> good.test()
```

Use models to predict a randomly selected image's label:

```py
>>> bad.predict()
>>> good.predict()
```

Train models:

```py
>>> bad.train()
>>> good.train()
```

Overwrite saved variables:

```py
>>> bad.save()
>>> good.save()
```

More information about Make targets:

```
$ make help
```
