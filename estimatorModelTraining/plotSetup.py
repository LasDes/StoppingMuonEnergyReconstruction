def setupRcParams(rcParams, grid=False):
    rcParams["font.family"] = "Akkurat"
    rcParams["xtick.major.size"] = 6
    rcParams["ytick.major.size"] = 6
    rcParams["xtick.major.width"] = 1
    rcParams["ytick.major.width"] = 1
    rcParams["xtick.minor.size"] = 3
    rcParams["ytick.minor.size"] = 3
    rcParams["xtick.minor.width"] = 1
    rcParams["ytick.minor.width"] = 1
    rcParams["axes.linewidth"] = 1
    rcParams["legend.fontsize"] = "medium"
    rcParams["legend.labelspacing"] = 0.8
    rcParams["legend.borderaxespad"] = 1.0
    rcParams["axes.grid"] = grid
    rcParams["grid.linestyle"] = "-"
    rcParams["grid.alpha"] = 0.1
    rcParams["grid.linewidth"] = 1.0
    rcParams["legend.frameon"] = False
    rcParams["xtick.minor.visible"] = True
    rcParams["ytick.minor.visible"] = True
    rcParams["lines.linewidth"] = 1.0

COLORS = {
	"r":        "#f05a30",
	"r_dark":   "#a61c20",
	"r_light":  "#f7955a",
	"b":        "#0069b5",
	"b_dark":   "#28256a",
	"b_light":  "#7aaede",
	"g":        "#7ac142",
	"g_dark":   "#4c8438",
	"g_light":  "#bed85e",
	"y":        "#e8c31d",
	"y_dark":   "#d09b2c",
	"y_light":  "#ffde4f",
	"m":        "#8151a1",
	"m_dark":   "#621856",
	"m_light":  "#bc8cbe"
}