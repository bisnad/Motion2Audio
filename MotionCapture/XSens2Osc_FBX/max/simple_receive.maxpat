{
	"patcher" : 	{
		"fileversion" : 1,
		"appversion" : 		{
			"major" : 8,
			"minor" : 5,
			"revision" : 4,
			"architecture" : "x64",
			"modernui" : 1
		}
,
		"classnamespace" : "box",
		"rect" : [ 75.0, 449.0, 1804.0, 804.0 ],
		"bglocked" : 0,
		"openinpresentation" : 0,
		"default_fontsize" : 12.0,
		"default_fontface" : 0,
		"default_fontname" : "Arial",
		"gridonopen" : 1,
		"gridsize" : [ 15.0, 15.0 ],
		"gridsnaponopen" : 1,
		"objectsnaponopen" : 1,
		"statusbarvisible" : 2,
		"toolbarvisible" : 1,
		"lefttoolbarpinned" : 0,
		"toptoolbarpinned" : 0,
		"righttoolbarpinned" : 0,
		"bottomtoolbarpinned" : 0,
		"toolbars_unpinned_last_save" : 0,
		"tallnewobj" : 0,
		"boxanimatetime" : 200,
		"enablehscroll" : 1,
		"enablevscroll" : 1,
		"devicewidth" : 0.0,
		"description" : "",
		"digest" : "",
		"tags" : "",
		"style" : "",
		"subpatcher_template" : "",
		"assistshowspatchername" : 0,
		"boxes" : [ 			{
				"box" : 				{
					"id" : "obj-44",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 1563.0, 108.0, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-45",
					"linecount" : 25,
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1563.0, 208.0, 159.0, 353.0 ],
					"presentation_linecount" : 25,
					"text" : "-0.503174 -0.0271 -0.608643 -0.756592 0.464111 0.19043 -0.0896 0.803955 -0.220947 -0.057617 0.592285 -0.6521 -0.877686 0.112305 -0.134521 -0.77832 -0.125977 -0.260742 0.186279 -0.428223 -0.66748 0.29541 -0.435547 -0.61084 -0.554443 0.619385 -0.350342 -0.688965 0.326416 -0.101807 0.588867 0.549805 -0.019775 -0.584961 -0.520264 0.310303 -0.383789 0.38916 0.603271 -0.77417 0.012939 -0.420898 -0.461426 0.541748 0.334717 -0.5354 -0.472412 0.431641 -0.812012 0.093994 -0.345703"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-46",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"patching_rect" : [ 1557.0, 69.0, 162.0, 22.0 ],
					"text" : "route /mocap/tracker/magnet"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-47",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1563.0, 151.0, 42.0, 22.0 ],
					"text" : "gate 1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-40",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 1385.0, 108.0, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-41",
					"linecount" : 34,
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1385.0, 208.0, 159.0, 478.0 ],
					"presentation_linecount" : 34,
					"text" : "0.756067 -0.023256 -0.653148 0.034921 0.218275 -0.523082 0.093621 -0.818521 0.299134 -0.625155 -0.668106 0.270804 0.46496 -0.475847 -0.484873 0.567698 0.102922 0.037661 -0.629895 0.768909 0.04115 -0.167435 0.048031 0.983852 0.622139 -0.450409 -0.284271 0.573816 0.698084 0.454707 -0.34583 -0.43165 0.659503 -0.358013 -0.554256 -0.360114 0.491279 -0.364421 -0.573268 -0.545166 0.753016 -0.657758 -0.003401 0.01762 0.364338 0.639224 -0.375275 0.563755 0.300708 -0.758488 -0.19181 -0.545417 0.108027 0.20762 -0.117287 0.965126 0.404082 -0.665857 -0.356036 -0.516323 0.297125 0.689636 -0.439327 0.493063 0.034916 -0.223905 0.001042 -0.973985"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-42",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"patching_rect" : [ 1379.0, 69.0, 171.0, 22.0 ],
					"text" : "route /mocap/tracker/rot_world"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-43",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1385.0, 151.0, 42.0, 22.0 ],
					"text" : "gate 1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-36",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 1220.0, 108.0, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-37",
					"linecount" : 3,
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1220.0, 208.0, 159.0, 50.0 ],
					"presentation_linecount" : 3,
					"text" : "1. 5. 7. 8. 9. 10. 11. 12. 13. 14. 15. 16. 17. 18. 20. 21. 22."
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-38",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"patching_rect" : [ 1214.0, 69.0, 131.0, 22.0 ],
					"text" : "route /mocap/tracker/id"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-39",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1220.0, 151.0, 42.0, 22.0 ],
					"text" : "gate 1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-32",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 1049.0, 108.0, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-33",
					"linecount" : 34,
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1049.0, 208.0, 157.0, 478.0 ],
					"presentation_linecount" : 34,
					"text" : "-0.293315 -0.369858 0.50477 -0.33954 0.25617 0.380391 -0.360084 0.534405 0.325112 -0.380629 0.81264 0.269833 -0.396029 1.021319 0.228373 -0.271532 1.071166 -0.33363 -0.040307 1.163745 -1.377347 -1.612537 4.017752 -1.221883 -0.213236 1.29579 -4.127672 -0.045097 2.22441 -3.272608 8.209717 5.076405 -3.749474 0.570574 -1.984846 0.194648 -0.610361 0.609089 -1.514939 0.108054 0.001639 0.496075 1.037187 0.176557 1.676033 0.191538 0.557213 0.070471 0.071867 -0.091255 1.294247 0.270927 0.097558 -0.165465 0.270927 0.097558 -0.165465 0.021432 -0.434226 0.478958 -0.194561 -0.30123 -0.458665 0.31516 -0.067894 -0.06691 0.31516 -0.067894 -0.06691"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-34",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"patching_rect" : [ 1049.0, 69.0, 147.0, 22.0 ],
					"text" : "route /mocap/joint/rot_acc"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-35",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 1049.0, 151.0, 42.0, 22.0 ],
					"text" : "gate 1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-31",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 885.0, 112.0, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-28",
					"linecount" : 34,
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 212.0, 157.0, 478.0 ],
					"presentation_linecount" : 34,
					"text" : "0.001955 0.025241 0.002358 0.001124 -0.020161 -0.023921 0.000755 -0.04034 -0.035601 0.000385 -0.060519 -0.04728 0.000108 -0.075653 -0.05604 0.020169 -0.060643 -0.038136 0.057425 -0.032768 -0.004886 0.021097 -0.059673 0.005801 -0.032136 -0.411262 -0.067664 0.015282 -0.618457 -0.287803 0.073807 -1.534873 -0.403974 -0.015064 -0.168557 -0.045121 -0.036341 -0.033848 -0.038661 -0.034222 -0.064717 -0.028303 -0.03127 -0.072898 -0.026222 0.0444 0.01063 0.067886 -0.002723 0.020762 -0.000845 -0.019373 0.00903 -0.001636 -0.019373 0.00903 -0.001636 -0.011277 0.045757 -0.012554 -0.015256 0.017364 -0.022101 -0.02235 0.006773 -0.002793 -0.02235 0.006773 -0.002793"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-29",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"patching_rect" : [ 879.0, 69.0, 143.0, 22.0 ],
					"text" : "route /mocap/joint/rot_vel"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-30",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 885.0, 155.0, 42.0, 22.0 ],
					"text" : "gate 1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-24",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 721.0, 119.0, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-25",
					"linecount" : 35,
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 721.0, 212.0, 157.0, 491.0 ],
					"presentation_linecount" : 35,
					"text" : "0.0588 -0.041199 -0.004207 -0.05767 0.079669 0.105253 -0.106717 0.133369 0.153885 -0.155763 0.187069 0.202516 -0.184219 0.236173 0.23902 -0.125944 0.189926 0.174781 -0.006772 0.120354 0.055545 -0.275724 0.129638 0.029022 -0.303659 -0.107438 0.214814 -0.671413 0.087889 0.350177 -0.948644 0.096691 0.687922 -0.255174 0.076157 0.11392 0.158606 -0.027607 -0.061044 0.114648 -0.097353 -0.120986 -0.141301 -0.14414 -0.175241 0.066308 -0.030592 0.019781 -0.018221 -0.051015 -0.002483 0.001737 0.007364 0.005813 0.000625 0.007351 0.005813 0.132461 0.037548 0.021192 0.082416 -0.033733 -0.002139 -0.009043 0.007114 -0.008916 -0.011202 0.007673 -0.008918"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-26",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"patching_rect" : [ 715.0, 69.0, 145.0, 22.0 ],
					"text" : "route /mocap/joint/lin_acc"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-27",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 721.0, 155.0, 42.0, 22.0 ],
					"text" : "gate 1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-20",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 557.0, 119.0, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-21",
					"linecount" : 34,
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 557.0, 212.0, 157.0, 478.0 ],
					"presentation_linecount" : 34,
					"text" : "0.011482 -0.004845 0.000763 0.011522 -0.003426 -0.000717 0.011533 0.000331 -0.000856 0.011494 0.003535 -0.000277 0.011496 -0.010992 0.00801 0.046281 0.016758 0.000475 0.011488 -0.001616 0.002079 0.046276 -0.007851 0.007172 0.046169 -0.00959 0.000923 0.186073 -0.078904 0.200874 0.185816 -0.076493 0.333913 0.011526 -0.005234 -0.002604 0.002858 -0.018545 -0.013337 -0.022986 -0.003981 -0.012572 -0.023058 -0.00435 -0.01342 0.011493 -0.000829 -0.00015 0.002863 -0.003708 0.001114 0.000712 -0.000292 0.000351 0.000712 -0.000292 0.000351 -0.001424 -0.0031 0.000517 0.002868 -0.000648 0.000461 0.000713 -0.000931 0.001126 0.000713 -0.000931 0.001126"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-22",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"patching_rect" : [ 551.0, 69.0, 141.0, 22.0 ],
					"text" : "route /mocap/joint/lin_vel"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-23",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 557.0, 155.0, 42.0, 22.0 ],
					"text" : "gate 1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-16",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 362.0, 119.0, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-17",
					"linecount" : 32,
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 362.0, 212.0, 157.0, 450.0 ],
					"presentation_linecount" : 32,
					"text" : "1.946286 -5.270581 -0.138948 -0.036277 -3.872144 0.042217 -6.741333 -3.248782 0.107066 -7.624759 -2.624501 0.162301 -2.45 -2.155777 0.197428 9.802224 -4.067042 -0.297979 -3.432 -7.624311 -1.19075 -6.597055 -8.443178 6.331716 33.572853 15.640829 78.68441 60.875179 10.745204 89.720779 62.993294 -13.225422 85.664207 -6.177021 12.432279 -13.246577 0.657714 -4.42311 -88.722656 25.163143 -6.489754 -86.667496 13.204585 -3.666278 -92.453499 2.59 -7.044104 0.412176 3.331685 -4.314558 0.110761 0.807525 -10.088435 -0.010341 0. -10.088508 0. 1.574582 -7.12914 0.535271 6.009631 -5.714824 0.957469 0.959776 -8.023311 -1.648276 0.959776 -8.023311 -1.648276"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-18",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"patching_rect" : [ 356.0, 69.0, 190.0, 22.0 ],
					"text" : "route /mocap/joint/rot_world_euler"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-19",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 362.0, 155.0, 42.0, 22.0 ],
					"text" : "gate 1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-15",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 197.0, 119.0, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-12",
					"linecount" : 45,
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 197.0, 212.0, 157.0, 629.0 ],
					"presentation_linecount" : 45,
					"text" : "0.998875 0.002098 0.011744 -0.045899 0.99858 0.005803 0.000895 -0.052952 0.996901 0.004589 -0.054663 -0.05639 0.996428 0.006071 -0.059519 -0.059599 0.997973 0.01033 -0.012366 -0.061559 0.993617 0.009549 0.090406 -0.066795 0.996326 -0.013026 -0.032063 -0.078334 0.989802 0.054489 -0.014026 -0.130865 0.708653 0.705071 0.025022 0.007716 0.724285 0.686531 -0.000275 0.063922 0.721634 0.690079 0.046373 0.02975 0.993956 -0.109593 0.003366 -0.00539 0.711722 -0.698616 0.035015 -0.064503 0.717015 -0.656706 0.130689 -0.19377 0.690636 -0.712251 0.064506 -0.107512 0.998145 0.00747 0.019978 -0.057022 0.999077 0.004315 0.027101 -0.033032 0.996104 0.000034 0.006925 -0.087915 0.996128 0. 0. -0.087917 0.997597 0.00645 0.011456 -0.068026 0.997434 0.013094 0.049733 -0.049812 0.997451 -0.013216 0.009612 -0.069452 0.997451 -0.013216 0.009612 -0.069452"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-13",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"patching_rect" : [ 191.0, 69.0, 157.0, 22.0 ],
					"text" : "route /mocap/joint/rot_world"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-14",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 197.0, 155.0, 42.0, 22.0 ],
					"text" : "gate 1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-11",
					"linecount" : 34,
					"maxclass" : "message",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 26.0, 212.0, 156.0, 478.0 ],
					"text" : "0.048549 0.037791 0.874388 0.037865 0.041164 0.968044 0.032739 0.04171 1.071501 0.016738 0.043437 1.164697 -0.001113 0.044709 1.257457 -0.014946 0.043539 1.389158 -0.002252 0.038759 1.476067 -0.014048 0.015685 1.330517 -0.042854 -0.115894 1.315421 -0.017822 -0.125826 1.025813 0.010538 -0.149394 0.791022 -0.003573 0.072494 1.332122 0.007198 0.202963 1.297007 0.039963 0.203555 1.008001 0.06949 0.22432 0.773089 0.035421 -0.038311 0.875751 0.010061 -0.035025 0.474803 -0.02461 -0.035436 0.08478 0.120488 -0.064182 0.01605 0.061654 0.1139 0.873221 0.037318 0.113033 0.472197 -0.008646 0.120704 0.08342 0.137202 0.100513 0.014043"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-9",
					"maxclass" : "toggle",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "int" ],
					"parameter_enable" : 0,
					"patching_rect" : [ 26.0, 119.0, 24.0, 24.0 ]
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-5",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 2,
					"outlettype" : [ "", "" ],
					"patching_rect" : [ 20.0, 69.0, 162.0, 22.0 ],
					"text" : "route /mocap/joint/pos_world"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-4",
					"maxclass" : "newobj",
					"numinlets" : 2,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 26.0, 155.0, 42.0, 22.0 ],
					"text" : "gate 1"
				}

			}
, 			{
				"box" : 				{
					"id" : "obj-1",
					"maxclass" : "newobj",
					"numinlets" : 1,
					"numoutlets" : 1,
					"outlettype" : [ "" ],
					"patching_rect" : [ 20.0, 21.0, 97.0, 22.0 ],
					"text" : "udpreceive 9004"
				}

			}
 ],
		"lines" : [ 			{
				"patchline" : 				{
					"destination" : [ "obj-13", 0 ],
					"order" : 8,
					"source" : [ "obj-1", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-18", 0 ],
					"order" : 7,
					"source" : [ "obj-1", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-22", 0 ],
					"order" : 6,
					"source" : [ "obj-1", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-26", 0 ],
					"order" : 5,
					"source" : [ "obj-1", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-29", 0 ],
					"order" : 4,
					"source" : [ "obj-1", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-34", 0 ],
					"order" : 3,
					"source" : [ "obj-1", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-38", 0 ],
					"order" : 2,
					"source" : [ "obj-1", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-42", 0 ],
					"order" : 1,
					"source" : [ "obj-1", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-46", 0 ],
					"order" : 0,
					"source" : [ "obj-1", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-5", 0 ],
					"order" : 9,
					"source" : [ "obj-1", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-14", 1 ],
					"source" : [ "obj-13", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-12", 1 ],
					"source" : [ "obj-14", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-14", 0 ],
					"source" : [ "obj-15", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-19", 0 ],
					"source" : [ "obj-16", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-19", 1 ],
					"source" : [ "obj-18", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-17", 1 ],
					"source" : [ "obj-19", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-23", 0 ],
					"source" : [ "obj-20", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-23", 1 ],
					"source" : [ "obj-22", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-21", 1 ],
					"source" : [ "obj-23", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-27", 0 ],
					"source" : [ "obj-24", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-27", 1 ],
					"source" : [ "obj-26", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-25", 1 ],
					"source" : [ "obj-27", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-30", 1 ],
					"source" : [ "obj-29", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-28", 1 ],
					"source" : [ "obj-30", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-30", 0 ],
					"source" : [ "obj-31", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-35", 0 ],
					"source" : [ "obj-32", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-35", 1 ],
					"source" : [ "obj-34", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-33", 1 ],
					"source" : [ "obj-35", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-39", 0 ],
					"source" : [ "obj-36", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-39", 1 ],
					"source" : [ "obj-38", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-37", 1 ],
					"source" : [ "obj-39", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-11", 1 ],
					"source" : [ "obj-4", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-43", 0 ],
					"source" : [ "obj-40", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-43", 1 ],
					"source" : [ "obj-42", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-41", 1 ],
					"source" : [ "obj-43", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-47", 0 ],
					"source" : [ "obj-44", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-47", 1 ],
					"source" : [ "obj-46", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-45", 1 ],
					"source" : [ "obj-47", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4", 1 ],
					"source" : [ "obj-5", 0 ]
				}

			}
, 			{
				"patchline" : 				{
					"destination" : [ "obj-4", 0 ],
					"source" : [ "obj-9", 0 ]
				}

			}
 ],
		"dependency_cache" : [  ],
		"autosave" : 0
	}

}
