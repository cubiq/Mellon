from PIL import Image

def resize(image: Image.Image | str | list[Image.Image | str], width: int, height: int, resample: int | str = Image.Resampling.BICUBIC) -> Image.Image | list[Image.Image]:
    """
    Resize an image or a list of images to the given width and height.
    """

    if isinstance(image, list):
        return [resize(i, width, height, resample) for i in image] if image else []

    if isinstance(image, str):
        image = Image.open(image)

    if isinstance(resample, str):
        resample = resample.upper()
        resample = Image.Resampling[resample] if resample in Image.Resampling else Image.Resampling.BICUBIC

    return image.resize((max(width, 1), max(height, 1)), resample=resample)


def cover(image: Image.Image | str | list[Image.Image | str], width: int, height: int, resample: int | str = Image.Resampling.BICUBIC) -> Image.Image | list[Image.Image]:
    """
    Resize an image or a list of images to cover the given width and height keeping the aspect ratio.
    """

    from PIL.ImageOps import cover as PILCover

    if isinstance(image, list):
        return [cover(i, width, height, resample) for i in image] if image else []

    if isinstance(image, str):
        image = Image.open(image)

    if isinstance(resample, str):
        resample = resample.upper()
        resample = Image.Resampling[resample] if resample in Image.Resampling else Image.Resampling.BICUBIC

    return PILCover(image, (max(width, 1), max(height, 1)), method=resample)

def fit(image: Image.Image | str | list[Image.Image | str], width: int, height: int, resample: int | str = Image.Resampling.BICUBIC) -> Image.Image | list[Image.Image]:
    """
    Resize an image or a list of images to fit the given width and height.
    The image will be resized to fit the given width and height, but the aspect ratio will be preserved.
    """

    from PIL.ImageOps import fit as PILFit

    if isinstance(image, list):
        return [fit(i, width, height, resample) for i in image] if image else []

    if isinstance(image, str):
        image = Image.open(image)

    if isinstance(resample, str):
        resample = resample.upper()
        resample = Image.Resampling[resample] if resample in Image.Resampling else Image.Resampling.BICUBIC

    return PILFit(image, (max(width, 1), max(height, 1)), method=resample)

def contain(image: Image.Image | str | list[Image.Image | str], width: int, height: int, resample: int | str = Image.Resampling.BICUBIC) -> Image.Image | list[Image.Image]:
    """
    Resize an image or a list of images to contain the given width and height keeping the aspect ratio.
    The image will be resized to the smallest possible size that contains the given width and height.
    Equivalent to PIL.thumbnail but doesn't modify the image object in place.
    """

    from PIL.ImageOps import contain as PILContain

    if isinstance(image, list):
        return [contain(i, width, height, resample) for i in image] if image else []

    if isinstance(image, str):
        image = Image.open(image)

    if isinstance(resample, str):
        resample = resample.upper()
        resample = Image.Resampling[resample] if resample in Image.Resampling else Image.Resampling.BICUBIC

    return PILContain(image, (max(width, 1), max(height, 1)), method=resample)


def pad(image: Image.Image | str | list[Image.Image | str], width: int, height: int, resample: int | str = Image.Resampling.BICUBIC, color: str | int | tuple[int, int, int] | None = (0, 0, 0)) -> Image.Image | list[Image.Image]:
    """
    Resize an image or a list of images keeping the aspect ratio and fit the provided width/height,
    the excess space will be filled with the given color.
    """

    from PIL.ImageOps import pad as PILPad

    if isinstance(image, list):
        return [pad(i, width, height, resample, color) for i in image] if image else []

    if isinstance(image, str):
        image = Image.open(image)

    if isinstance(resample, str):
        resample = resample.upper()
        resample = Image.Resampling[resample] if resample in Image.Resampling else Image.Resampling.BICUBIC

    return PILPad(image, (max(width, 1), max(height, 1)), method=resample, color=color)