# This file is part of Zennit
# Copyright (C) 2019-2021 Christopher J. Anders
#
# zennit/image.py
#
# Zennit is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option) any
# later version.
#
# Zennit is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
# more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library. If not, see <https://www.gnu.org/licenses/>.
"""Functionality to convert arrays to images"""
import numpy as np
from PIL import Image
import re
from typing import NamedTuple


class CMapToken(NamedTuple):
    """Tokens used by the lexer of ColorMap."""

    type: str
    value: str
    pos: int


class ColorNode(NamedTuple):
    """Nodes produced by the parser of ColorMap."""

    index: int
    value: np.ndarray


class ColorMap:
    """Compile a color map from color-map specification language (cmsl) source code.

    The color-map specification language (cmsl) is used to specify linear color maps with comma-separated colors
    supplied as hexadecimal values for each color channel in RGB, with either 1 or 2 values per channel.  Optionally, a
    hexadecimal index with either one or two digits may be supplied in front of each color, followed by a colon, to
    indicate the index which should be the color. Values for the ColorMap in-between colors will be interpolated
    linearly. If no index is supplied, colors without indices will be spaced evenly between indices. If the first and
    last indices are supplied but not 0 (or 00) and f (or ff) respectively, they will be added as an additional node in
    the color map, with the same color as the colors with the lowest and highest index respectively.  If indices are
    provided, they must be in ascending order from left to right, with an arbitrary number of non-indexed colors. If
    the first and/or last color are not indexed, they are assumed to be 0 (or 00) and f (or ff) respectively.

    Parameters
    ----------
    source : str
        Source code to generate the color map.

    """

    _rexp = re.compile(
        r"(?P<longcolor>[0-9a-fA-F]{6})|"
        r"(?P<shortcolor>[0-9a-fA-F]{3})|"
        r"(?P<index>[0-9a-fA-F]{1,2})|"
        r"(?P<adsep>:)|"
        r"(?P<sep>,)|"
        r"(?P<whitespace>\s+)|"
        r"(?P<error>.+)"
    )

    def __init__(self, source):
        self._source = None
        self.source = source

    @property
    def source(self) -> str:
        """Source code property used to generate the color map. May be overwritten with a new string, which will be
        compiled to change the color map.
        """
        return self._source

    @source.setter
    def source(self, value):
        """Set source code property and re-compile the color map.

        Parameters
        ----------
        value : str
            The code for the color map.

        Raises
        ------
        RuntimeError
            If the compilation failed, usually due to errors in the code.

        """
        try:
            tokens = self._lex(value)
            nodes = self._parse(tokens)
            self._indices, self._colors = self._make_palette(nodes)
        except RuntimeError as err:
            raise RuntimeError("Compilation of ColorMap failed!") from err

        self._source = value

    @staticmethod
    def _lex(string):
        """Lexical scanning of cmsl using regular expressions.

        Parameters
        ----------
        string : str
            String to scan.

        Returns
        -------
        list of `CMapToken`
            The resulting tokens.

        """
        return [
            CMapToken(match.lastgroup, match.group(), match.start())
            for match in ColorMap._rexp.finditer(string)
        ]

    @staticmethod
    def _parse(tokens):
        """Parse cmsl tokens into a list of color nodes.

        Parameters
        ----------
        tokens : list of `CMapToken
            A list of scanned cmsl tokens.

        Returns
        -------
        list of `ColorNode`
            The identified color nodes.

        Raises
        ------
        RuntimeError
            If there was an unexpected token.
        """
        nodes = []
        log = []
        for token in tokens:
            if token.type == "index" and not log:
                log.append(token)
            elif token.type == "adsep" and len(log) == 1 and log[-1].type == "index":
                log.append(token)
            elif token.type in ("shortcolor", "longcolor"):
                if len(log) == 2 and log[-2].type == "index":
                    indval = log[-2].value
                    if len(indval) == 1:
                        indval = indval * 2
                    index = int(indval, base=16)
                elif not log:
                    index = None
                else:
                    raise RuntimeError(f"Unexpected {token}")

                value_it = (
                    iter(token.value) if token.type == "longcolor" else token.value
                )
                value = [int("".join(chars), base=16) for chars in zip(*[value_it] * 2)]
                nodes.append(ColorNode(index, np.array(value)))
                log.append(token)
            elif (
                token.type == "sep"
                and log
                and log[-1].type in ("longcolor", "shortcolor")
            ):
                log.clear()
            elif token.type != "whitespace":
                raise RuntimeError(f"Unexpected {token}")

        if log and log[-1].type not in ("shortcolor", "longcolor"):
            raise RuntimeError(f"Unexpected {log[-1]}")

        return nodes

    @staticmethod
    def _make_palette(nodes):
        """Generate color map indices and colors from a list of color nodes.

        Parameters
        ----------
        nodes : list of `ColorNode`
            The color nodes which identify the color map.

        Returns
        -------
        indices : :py:obj:`numpy.ndarray`
            An array of shape N x 3, where 3 is the number of color nodes, containing the RGB colors.
        colors : :py:obj:`numpy.ndarray`
            An array of shape N of type ``numpy.uint8``, which are the indices of the color nodes.

        Raises
        ------
        RuntimeError
            If there are less than 2 colors in the color map.

        """
        if len(nodes) < 2:
            raise RuntimeError("ColorMap needs at least 2 colors!")
        result = []
        log = []

        start = nodes.pop(0)
        result.append(ColorNode(0, start.value))
        if start.index is not None and start.index > 0:
            result.append(start)

        for n, node in enumerate(nodes):
            if node.index is None:
                if n < len(nodes) - 1:
                    log.append(node)
                    continue
                node = ColorNode(255, node.value)
            elif node.index < result[-1].index:
                raise RuntimeError(
                    "ColorMap indices not ordered! Provided indices are required in ascending order."
                )
            if log:
                result += [
                    ColorNode(
                        int(result[-1].index * (1.0 - alpha) + node.index * alpha),
                        lognode.value,
                    )
                    for alpha, lognode in zip(
                        np.linspace(0.0, 1.0, len(log) + 2)[1:-1], log
                    )
                ]
                log.clear()
            result.append(node)

        result.append(ColorNode(256, result[-1].value))

        indices = np.array([node.index for node in result])
        colors = np.stack([node.value for node in result], axis=0)

        return indices, colors

    def __call__(self, x):
        """Map scalar values in the range [0, 1] to RGB. This appends an axis with size 3 to `x`. Values are clipped to
        the range [0, 1], and the output will also lie in this range.

        Parameters
        ----------
        x : obj:`numpy.ndarray`
            Input array of arbitrary shape, which will be clipped to range [0, 1], and mapped to RGB using this
            ColorMap.

        Returns
        -------
        obj:`numpy.ndarray`
            The input array `x`, clipped to [0, 1] and mapped to RGB given this colormap, where the 3 color channels
            are appended as a new axis to the end.
        """
        x = (x * 255).clip(0, 255)
        index = np.searchsorted(self._indices[:-1], x, side="right")
        alpha = (
            (x - self._indices[index - 1])
            / (self._indices[index] - self._indices[index - 1])
        )[..., None]
        return (
            self._colors[index - 1] * (1 - alpha) + self._colors[index] * alpha
        ) / 255.0

    def palette(self, level=1.0):
        """Create an 8-bit palette.

        Parameters
        ----------
        level: float
            The level of the color map. 1.0 is default. Values below zero reduce the color range, with only a single
            color used at value 0.0. Values above 1.0 clip the value earlier towards the maximum, with an increasingly
            steep transition at the center of the image.

        Returns
        -------
        obj:`numpy.ndarray`
            The palette described by an unsigned 8-bit numpy array with 256 entries.
        """
        x = np.linspace(-1.0, 1.0, 256, dtype=np.float64) * level
        x = ((x + 1.0) / 2.0).clip(0.0, 1.0)
        x = self(x)
        x = (x * 255.0).round(12).clip(0.0, 255.0).astype(np.uint8)
        return x


class LazyColorMapCache:
    """Dict-like object to store sources for colormaps, and compile and cache them lazily.

    Parameters
    ----------
    sources : dict
        Dict containing a mapping from names to color map specification language source.
    """

    def __init__(self, sources):
        self._sources = sources
        self._compiled = {}

    def __getitem__(self, name):
        if name not in self._sources:
            raise KeyError(f"No source for key {name}.")
        if name not in self._compiled:
            self._compiled[name] = ColorMap(self._sources[name])
        return self._compiled[name]

    def __setitem__(self, name, value):
        self._sources[name] = value
        if name in self._compiled:
            self._compiled[name].source = value

    def __delitem__(self, name):
        del self._sources[name]
        if name in self._compiled:
            del self._compiled[name]

    def __iter__(self):
        return iter(self._sources)

    def __len__(self):
        return len(self._sources)


# CMAPS contains all built-in color maps
CMAPS = LazyColorMapCache(
    {
        # black to white
        "gray": "000,fff",
        # white to red
        "wred": "fff,f00",
        # white to blue
        "wblue": "fff,00f",
        # black to red to yellow to white
        "hot": "000,f00,ff0,fff",
        # black to blue to cyan
        "cold": "000,00f,0ff",
        # combination of cold (reversed) and hot, centered around black
        "coldnhot": "0ff,00f,80:000,f00,ff0,fff",
        # combination of wblue (reversed) and wred, centered around white
        "bwr": "00f,80:fff,f00",
        # blue to white to red as in the french flag
        "france": "0055a4,80:fff,ef4135",
        # blue to white to red with brightness 0xd0
        "seismic": "0000d0,80:d0d0d0,d00000",
        # cyan to white to magenta with brightness 0xd0
        "coolio": "00d0d0,80:d0d0d0,d000d0",
        # green to white to magenta with brightness 0xd0
        "coleus": "00d000,80:d0d0d0,d000d0",
    }
)


def get_cmap(cmap):
    """Convenience function to lookup built-in color maps, or create color maps from a source code.

    Parameters
    ----------
    cmap : str or ColorMap
        String to specify a built-in color map, code used to create a new color map, or a ColorMap instance.

    Returns
    -------
    ColorMap
        The built-in color map with key `cmap` in CMAPS, a new color map created from the code `cmap`, or `cmap` if it
        already was a ColorMap.
    """
    if isinstance(cmap, ColorMap):
        return cmap
    if cmap in CMAPS:
        return CMAPS[cmap]
    return ColorMap(cmap)


def palette(cmap="bwr", level=1.0):
    """Convenience function to create palettes from built-in colormaps, or from a source code if necessary.

    Parameters
    ----------
    cmap: str or ColorMap
        String to specify a built-in color map, code used to create a new color map, or a ColorMap instance, which will
        be used to create a palette.
    level: float
        The level of the color map palette. 1.0 is default. Values below zero reduce the color range, with only a
        single color used at value 0.0. Values above 1.0 clip the value earlier towards the maximum, with an
        increasingly steep transition at the center of the color map range.

    Returns
    -------
    obj:`numpy.ndarray`
        The palette described by an unsigned 8-bit numpy array with 256 entries.
    """
    colormap = get_cmap(cmap)
    return colormap.palette(level=level)


def imgify(
    obj,
    vmin=None,
    vmax=None,
    cmap="bwr",
    level=1.0,
    symmetric=False,
    grid=False,
    gridfill=None,
):
    """Convert an array with 1 or 3 channels to a PIL image.
    The color dimension can be either the first or the last dimension.

    Parameters
    ----------
    obj: object
        Anything that can be converted to a numpy array with 2 dimensions greyscale, or 3 dimensions with 1 or 3 values
        in the first or last dimension (color).
    vmin: float or obj:`numpy.ndarray`
        Manual minimum value of the array. Overrides the used norm's minimum value.
    vmax: float or obj:`numpy.ndarray`
        Manual maximum value of the array. Overrides the used norm's maximum value.
    cmap: str or ColorMap
        String to specify a built-in color map, code used to create a new color map, or a ColorMap instance, which will
        be used to create a palette. The color map will only be applied for arrays with only a single color channel.
        The color will be specified as a palette in the PIL Image.
    level: float
        The level of the color map. 1.0 is default. Values below 1.0 reduce the color range, with only a single color
        used at value 0.0. Values above 1.0 clip the value earlier towards the maximum, with an increasingly steep
        transition at the center of the pixel value distribution.
    symmetric : bool, optional
        Specifies whether the norm should be symmetric (True) or unaligned (False, default). If True, normalize with
        both minimum and maximum by the absolute maximum, which will cause 0. in the input to correspond to 0.5 in the
        result. Setting ``symmetric=False`` (default) will result in the minimum value to be directly mapped to 0 and
        the maximum value to be directly mapped to 1. ``vmin`` and ``vmax`` may be used to manually override the
        minimum and maximum value respectively.
    grid : bool or tuple of ints of size 2
        If true, assumes the first dimension to be the batch dimension. If True, creates a square grid of images in the
        batch dimension after normalizing each sample. If tuple of ints of size 2, creates the grid in the shape of
        ``(height, width)``. If False (default), does not assume a batch dimension.
    gridfill: :py:obj:`np.uint8`
        A value to fill empty grid members. Default is the mean pixel value. No effect when ``grid=False``.

    Returns
    -------
    image: obj:`PIL.Image`
        The array visualized as a Pillow Image.

    Raises
    ------
    TypeError
        If the shape of the array cannot be converted to an image.
    """
    array = np.array(obj)

    if grid:
        if isinstance(grid, (list, tuple)) and len(grid) != 2:
            raise TypeError("Grid shape needs to be of size 2!")

        if array.ndim not in (3, 4):
            raise TypeError("Grid input has to have either 3 or 4 axes!")

        if (array.ndim == 4) and (array.shape[3] not in (1, 3)):
            if array.shape[1] in (1, 3):
                array = array.transpose(0, 2, 3, 1)
            else:
                raise TypeError(
                    "After batch, last (or first) axis of input are color channels, "
                    "which have to either be 1, 3 or be omitted entirely!"
                )
    else:
        if array.ndim not in (2, 3):
            raise TypeError("Input has to have either 2 or 3 axes!")

        if (array.ndim == 3) and (array.shape[2] not in (1, 3)):
            if array.shape[0] in (1, 3):
                array = array.transpose(1, 2, 0)
            else:
                raise TypeError(
                    "Last (or first) axis of input are color channels, "
                    "which have to either be 1, 3 or be omitted entirely!"
                )

    # renormalize data if necessary
    if array.dtype != np.uint8:
        if grid:
            dims = tuple(range(1, array.ndim))
        else:
            dims = tuple(range(array.ndim))

        # initialize vmin_ and vmax_, so they are never undefined
        vmin_, vmax_ = None, None

        # only compute bounds for vmin_ and vmax_ if we really need to
        if None in (vmin, vmax):
            vmin_, vmax_ = interval_norm_bounds(array, symmetric=symmetric, dim=dims)

        if vmin is None:
            vmin = vmin_
        else:
            vmin = np.array(vmin)

        if vmax is None:
            vmax = vmax_
        else:
            vmax = np.array(vmax)

        array = (array - vmin) / (vmax - vmin)
        array = (array * 255).clip(0, 255).astype(np.uint8)

    if grid:
        shape = None if isinstance(grid, bool) else grid
        # gridify adds the missing axis
        array = gridify(array, shape=shape, fill_value=gridfill)
    else:
        # add missing axis if omitted
        if array.ndim == 2:
            array = array[:, :, None]

    # apply palette if single channel
    if array.shape[2] == 1:
        image = Image.fromarray(array[..., 0], mode="P")
        image.putpalette(palette(cmap, level))
    else:
        image = Image.fromarray(array, mode="RGB")

    return image


def gridify(obj, shape=None, fill_value=None):
    """Align multiple arrays, described as an additional 0-th dimension, into a grid with the 0-th dimension removed.

    Parameters
    ----------
    obj: object
        An object that can be converted to a numpy array, with 3 (greyscale) or 4 (rgb) axes.
        The color channel's position is automatically detected, and moved to the back of the shape.
    shape: tuple of size 2, optional
        Height and width of the produced grid. If None (default), create a square grid.
    fill_value: float or obj:`numpy.ndarray`
        A value to fill empty grid members. Default is the mean pixel value.

    Returns
    -------
    obj:`numpy.ndarray`
        An array with the 0-th dimension absorbed into the height and width dimensions (then 0 and 1).
        The color dimension will be the last dimension, even if it was the first dimension before.

    Raises
    ------
    TypeError
        If the shape of the array cannot be converted to an image.
    """
    array = np.array(obj)
    if array.ndim not in (3, 4):
        raise TypeError(
            "For creating an image grid, the array has to have either 3 (greyscale) or 4 (rgb) axes!"
        )

    # add missing axis if omitted
    if array.ndim == 3:
        array = array[..., None]

    if array.shape[3] not in (1, 3):
        if array.shape[1] in (1, 3):
            array = array.transpose(0, 2, 3, 1)
        else:
            raise TypeError(
                "Last (or first) axis of input are color channels, "
                "which have to either be 1, 3 or be omitted entirely!"
            )

    num, height, width, channels = array.shape

    if shape is None:
        grid_width = int(num**0.5)
        grid_height = (num + grid_width - 1) // grid_width
    else:
        grid_height, grid_width = shape

    if fill_value is None:
        fill_value = array.min((0, 1, 2), keepdims=True)
    else:
        fill_value = np.array(fill_value).astype(array.dtype)

    dim = min(num, grid_height * grid_width)
    result = (
        np.zeros((grid_height * grid_width, height, width, channels), dtype=array.dtype)
        + fill_value
    )
    result[:dim] = array[:dim]
    result = (
        result.reshape(grid_height, grid_width, height, width, channels)
        .transpose(0, 2, 1, 3, 4)
        .reshape(grid_height * height, grid_width * width, channels)
    )

    return result


def imsave(
    fp,
    obj,
    vmin=None,
    vmax=None,
    cmap="bwr",
    level=1.0,
    grid=False,
    format=None,
    writer_params=None,
    symmetric=False,
    gridfill=None,
):
    """Convert an array to an image and save it using file `fp`.
    Internally, `imgify` is called to create a PIL Image, which is then saved using PIL.

    Parameters
    ----------
    fp: str, obj:`pathlib.Path` or file
        Save target for PIL Image.
    obj: object
        Anything that can be converted to a numpy array with 2 dimensions greyscale, or 3 dimensions with 1 or 3 values
        in the first or last dimension (color).
    vmin: float or obj:`numpy.ndarray`
        Manual minimum value of the array. Overrides the used norm's minimum value.
    vmax: float or obj:`numpy.ndarray`
        Manual maximum value of the array. Overrides the used norm's maximum value.
    cmap: str or ColorMap
        String to specify a built-in color map, code used to create a new color map, or a ColorMap instance, which will
        be used to create a palette. The color map will only be applied for arrays with only a single color channel.
        The color will be specified as a palette in the PIL Image.
    level: float
        The level of the color map. 1.0 is default. Values below 1.0 reduce the color range, with only a single color
        used at value 0.0. Values above 1.0 clip the value earlier towards the maximum, with an increasingly steep
        transition at the center of the pixel value distribution.
    grid : bool or tuple of ints of size 2
        If true, assumes the first dimension to be the batch dimension. If True, creates a square grid of images in the
        batch dimension after normalizing each sample. If tuple of ints of size 2, creates the grid in the shape of
        ``(height, width)``. If False (default), does not assume a batch dimension.
    format: str
        Optional format override for PIL Image.save.
    writer_params: dict
        Extra params to the image writer in PIL.
    symmetric : bool, optional
        Specifies whether the norm should be symmetric (True) or unaligned (False, default). If True, normalize with
        both minimum and maximum by the absolute maximum, which will cause 0. in the input to correspond to 0.5 in the
        result. Setting ``symmetric=False`` (default) will result in the minimum value to be directly mapped to 0 and
        the maximum value to be directly mapped to 1. ``vmin`` and ``vmax`` may be used to manually override the
        minimum and maximum value respectively.
    gridfill: :py:obj:`np.uint8`
        A value to fill empty grid members. Default is the mean pixel value. No effect when ``grid=False``.
    """
    if writer_params is None:
        writer_params = {}
    image = imgify(
        obj,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        level=level,
        symmetric=symmetric,
        grid=grid,
        gridfill=gridfill,
    )
    image.save(fp, format=format, **writer_params)


def interval_norm_bounds(input, symmetric=False, dim=None):
    """Return the boundaries to normalize the data interval batch-wise between 0. and 1. given the specified strategy.

    Parameters
    ----------
    input : :py:class:`numpy.ndarray`
        Array for which to return the boundaries.
    symmetric : bool, optional
        Specifies whether the norm should be symmetric (True) or unaligned (False, default).
        If True, normalize with both minimum and maximum by the absolute maximum, which will cause 0. in the
        input to correspond to 0.5 in the result. Setting ``symmetric=False`` (default) will result in the minimum
        value to be directly mapped to 0 and the maximum value to be directly mapped to 1.
    dim : tuple of ints, optional
        Set the channel dimensions over which the boundaries are computed (default is ``tuple(range(1, input.ndim))``.

    Returns
    -------
    :py:class:`numpy.ndarray`
        The normalized array ``input`` along ``dim`` to lie in the interval [0, 1].

    """
    if dim is None:
        dim = tuple(range(1, input.ndim))

    if symmetric:
        # 0-aligned symmetric input, negative and positive can be compared, the original 0. becomes 0.5
        vmax = np.abs(input).max(dim, keepdims=True)
        vmin = -vmax
    else:
        # do not align, the original minimum value becomes 0., the original maximum becomes 1.
        vmax = input.max(dim, keepdims=True)
        vmin = input.min(dim, keepdims=True)
    return vmin, vmax
