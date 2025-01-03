import copy
import json
import uuid

import numpy as np


def family_instance_generate_node(
    transformation_dict, id2mesh, host_relationship
):
    root = {
        "Uuid": "fa59e632-6902-4b27-bd94-401999452ddd",
        "IsCustomNode": False,
        "Description": "",
        "Name": "a",
        "ElementResolver": {"ResolutionMap": {}},
        "Inputs": [],
        "Outputs": [],
        "Nodes": None,
        "Connectors": None,
        "Dependencies": [],
        "NodeLibraryDependencies": [],
        "Thumbnail": "",
        "GraphDocumentationURL": None,
        "ExtensionWorkspaceData": [
            {
                "ExtensionGuid": "28992e1d-abb9-417f-8b1b-05e053bee670",
                "Name": "特性",
                "Version": "2.19",
                "Data": {},
            },
            {
                "ExtensionGuid": "DFBD9CC0-DB40-457A-939E-8C8555555A9D",
                "Name": "Generative Design",
                "Version": "6.1",
                "Data": {},
            },
        ],
        "Author": "",
        "Linting": {
            "activeLinter": "无",
            "activeLinterId": "7b75fb44-43fd-4631-a878-29f4d5d8399a",
            "warningCount": 0,
            "errorCount": 0,
        },
        "Bindings": [],
        "View": {
            "Dynamo": {
                "ScaleFactor": 1.0,
                "HasRunWithoutCrash": True,
                "IsVisibleInDynamoLibrary": True,
                "Version": "2.19.3.6394",
                "RunType": "Manual",
                "RunPeriod": "1000",
            },
            "Camera": {
                "Name": "背景预览(_B)",
                "EyeX": -17.0,
                "EyeY": 24.0,
                "EyeZ": 50.0,
                "LookX": 12.0,
                "LookY": -13.0,
                "LookZ": -58.0,
                "UpX": 0.0,
                "UpY": 1.0,
                "UpZ": 0.0,
            },
            "ConnectorPins": [],
            "NodeViews": [],
            "Annotations": [],
            "X": 907.66938673484117,
            "Y": 456.59982075496794,
            "Zoom": 0.717662048161244,
        },
    }

    Node_family_template = {
        "ConcreteType": "DSRevitNodesUI.FamilyTypes, DSRevitNodesUI",
        "SelectedIndex": 0,
        "SelectedString": "Column2:Column2",
        "Id": "6c68b08e7c1b49dca1d68b3e8d8b9493",
        "NodeType": "ExtensionNode",
        "Inputs": [],
        "Outputs": [
            {
                "Id": "ab5d0227ad9943a5aab6df8612b4df9e",
                "Name": "Family Type",
                "Description": "选定的 Family Type",
                "UsingDefaultValue": False,
                "Level": 2,
                "UseLevels": False,
                "KeepListStructure": False,
            }
        ],
        "Replication": "Disabled",
        "Description": "文档中所有可用族类型。",
    }

    Node_level_template = [
        {
            "ConcreteType": "Dynamo.Graph.Nodes.ZeroTouch.DSFunction, DynamoCore",
            "Id": "1de135fcf9b54473aa1f801c84641a1f",
            "NodeType": "FunctionNode",
            "Inputs": [
                {
                    "Id": "5994a827c1924f2da2fe54e7e048efe2",
                    "Name": "elevation",
                    "Description": "double",
                    "UsingDefaultValue": False,
                    "Level": 2,
                    "UseLevels": False,
                    "KeepListStructure": False,
                }
            ],
            "Outputs": [
                {
                    "Id": "7ea795726aa14f5da79c043428d299f4",
                    "Name": "Level",
                    "Description": "Level",
                    "UsingDefaultValue": False,
                    "Level": 2,
                    "UseLevels": False,
                    "KeepListStructure": False,
                }
            ],
            "FunctionSignature": "Revit.Elements.Level.ByElevation@double",
            "Replication": "Auto",
            "Description": "根据项目中其高程和名称来创建 Revit 标高。名称将采用 Revit 所给的名称。\n\nLevel.ByElevation (elevation: double): Level",
        },
        {
            "ConcreteType": "CoreNodeModels.Input.DoubleInput, CoreNodeModels",
            "NumberType": "Double",
            "Id": "a5f3bccd427e45ab8bbed057ddcc765d",
            "NodeType": "NumberInputNode",
            "Inputs": [],
            "Outputs": [
                {
                    "Id": "cfb7a7b17c124f1d8291a415f5a3b1a7",
                    "Name": "",
                    "Description": "Double",
                    "UsingDefaultValue": False,
                    "Level": 2,
                    "UseLevels": False,
                    "KeepListStructure": False,
                }
            ],
            "Replication": "Disabled",
            "Description": "创建数字。",
            "InputValue": 0.0,
        },
    ]

    Node_list_template = [
        {
            "ConcreteType": "Dynamo.Graph.Nodes.CodeBlockNodeModel, DynamoCore",
            "Id": "6ffd4fe4f01b419bafb92846bcc1eabb",
            "NodeType": "CodeBlockNode",
            "Inputs": [],
            "Outputs": [
                {
                    "Id": "58452d6a33874b1c83cfda7cb78fafad",
                    "Name": "",
                    "Description": "行 1 中的表达式值",
                    "UsingDefaultValue": False,
                    "Level": 2,
                    "UseLevels": False,
                    "KeepListStructure": False,
                },
                {
                    "Id": "46e96c4b5f6244ee8343021daae31b8b",
                    "Name": "",
                    "Description": "行 2 中的表达式值",
                    "UsingDefaultValue": False,
                    "Level": 2,
                    "UseLevels": False,
                    "KeepListStructure": False,
                },
                {
                    "Id": "1429701dc32f4e24bf066f6f4d2124c9",
                    "Name": "",
                    "Description": "行 3 中的表达式值",
                    "UsingDefaultValue": False,
                    "Level": 2,
                    "UseLevels": False,
                    "KeepListStructure": False,
                },
                {
                    "Id": "7b0fe2bd502a4b1a916cb2ceb66e2ea3",
                    "Name": "",
                    "Description": "行 4 中的表达式值",
                    "UsingDefaultValue": False,
                    "Level": 2,
                    "UseLevels": False,
                    "KeepListStructure": False,
                },
            ],
            "Replication": "Disabled",
            "Description": "允许直接编写 DesignScript 代码",
            "Code": "10;\n0;\n0;\n30;",  # ! x, y, z, rz(degree)
        },
        {
            "ConcreteType": "Dynamo.Graph.Nodes.ZeroTouch.DSFunction, DynamoCore",
            "Id": "d31e765288cb48158684d152a41beb9a",
            "NodeType": "FunctionNode",
            "Inputs": [
                {
                    "Id": "30d179864c5246418fb2f69124463b65",
                    "Name": "x",
                    "Description": "X 轴坐标\n\ndouble\n默认值 : 0",
                    "UsingDefaultValue": True,
                    "Level": 2,
                    "UseLevels": False,
                    "KeepListStructure": False,
                },
                {
                    "Id": "23617247ea0a467db2880689259d44ff",
                    "Name": "y",
                    "Description": "Y 轴坐标\n\ndouble\n默认值 : 0",
                    "UsingDefaultValue": True,
                    "Level": 2,
                    "UseLevels": False,
                    "KeepListStructure": False,
                },
                {
                    "Id": "ba59a996b5d44c6292d22a4c0d268cf6",
                    "Name": "z",
                    "Description": "Z 轴坐标\n\ndouble\n默认值 : 0",
                    "UsingDefaultValue": True,
                    "Level": 2,
                    "UseLevels": False,
                    "KeepListStructure": False,
                },
            ],
            "Outputs": [
                {
                    "Id": "49f51eb983444badb462bd6b8a4068b8",
                    "Name": "Point",
                    "Description": "由坐标创建的点",
                    "UsingDefaultValue": False,
                    "Level": 2,
                    "UseLevels": False,
                    "KeepListStructure": False,
                }
            ],
            "FunctionSignature": "Autodesk.DesignScript.Geometry.Point.ByCoordinates@double,double,double",
            "Replication": "Auto",
            "Description": "通过给定的 3 个笛卡尔坐标形成一个点\n\nPoint.ByCoordinates (x: double = 0, y: double = 0, z: double = 0): Point",
        },
        {
            "ConcreteType": "Dynamo.Graph.Nodes.ZeroTouch.DSFunction, DynamoCore",
            "Id": "a5e851ffc1694a80a96721988458f021",
            "NodeType": "FunctionNode",
            "Inputs": [
                {
                    "Id": "c080a813e711475da79bd28e50fd84f6",
                    "Name": "familyType",
                    "Description": "族类型。也称为族符号。\n\nFamilyType",
                    "UsingDefaultValue": False,
                    "Level": 2,
                    "UseLevels": False,
                    "KeepListStructure": False,
                },
                {
                    "Id": "55791089ba5548749646958a7842f63e",
                    "Name": "point",
                    "Description": "以米为单位的点。\n\nPoint",
                    "UsingDefaultValue": False,
                    "Level": 2,
                    "UseLevels": False,
                    "KeepListStructure": False,
                },
                {
                    "Id": "6199c4780c3044ef812db759d8e9f428",
                    "Name": "level",
                    "Description": "主体族实例的标高。\n\nLevel",
                    "UsingDefaultValue": False,
                    "Level": 2,
                    "UseLevels": False,
                    "KeepListStructure": False,
                },
            ],
            "Outputs": [
                {
                    "Id": "6065416333634a059b54285277119ca5",
                    "Name": "FamilyInstance",
                    "Description": "FamilyInstance",
                    "UsingDefaultValue": False,
                    "Level": 2,
                    "UseLevels": False,
                    "KeepListStructure": False,
                }
            ],
            "FunctionSignature": "Revit.Elements.FamilyInstance.ByPointAndLevel@Revit.Elements.FamilyType,Autodesk.DesignScript.Geometry.Point,Revit.Elements.Level",
            "Replication": "Auto",
            "Description": "根据标高、FamilyType (在 Revit API 中也称为 FamilySymbol) 及其在世界空间的坐标，放置 Revit FamilyInstance\n\nFamilyInstance.ByPointAndLevel (familyType: FamilyType, point: Point, level: Level): FamilyInstance",
        },
        {
            "ConcreteType": "Dynamo.Graph.Nodes.ZeroTouch.DSFunction, DynamoCore",
            "Id": "6c1e2f13b7ea46d8a6e9a5a0cb77bd58",
            "NodeType": "FunctionNode",
            "Inputs": [
                {
                    "Id": "8d6daf8e6d9547d6a41a8e9c93962a30",
                    "Name": "familyInstance",
                    "Description": "Revit.Elements.FamilyInstance",
                    "UsingDefaultValue": False,
                    "Level": 2,
                    "UseLevels": False,
                    "KeepListStructure": False,
                },
                {
                    "Id": "6a1dc87305dc45c5a5647172ffab2d09",
                    "Name": "degree",
                    "Description": "绕 Z 轴的欧拉角。\n\ndouble",
                    "UsingDefaultValue": False,
                    "Level": 2,
                    "UseLevels": False,
                    "KeepListStructure": False,
                },
            ],
            "Outputs": [
                {
                    "Id": "5ecbeacb236749238ef8e31b30b095b5",
                    "Name": "FamilyInstance",
                    "Description": "结果族实例。",
                    "UsingDefaultValue": False,
                    "Level": 2,
                    "UseLevels": False,
                    "KeepListStructure": False,
                }
            ],
            "FunctionSignature": "Revit.Elements.FamilyInstance.SetRotation@double",
            "Replication": "Auto",
            "Description": "设置绕本地 Z 轴的族实例的欧拉角。\n\nFamilyInstance.SetRotation (degree: double): FamilyInstance",
        },
    ]

    Connector_list_template = [
        {
            "Start": "ab5d0227ad9943a5aab6df8612b4df9e",
            "End": "c080a813e711475da79bd28e50fd84f6",
            "Id": "74073d57ea32471a8ef757c7a59c5fc8",
            "IsHidden": "False",
        },
        {
            "Start": "58452d6a33874b1c83cfda7cb78fafad",
            "End": "30d179864c5246418fb2f69124463b65",
            "Id": "5693e2d625964e14988961b9a3e21446",
            "IsHidden": "False",
        },
        {
            "Start": "46e96c4b5f6244ee8343021daae31b8b",
            "End": "23617247ea0a467db2880689259d44ff",
            "Id": "dbd18a29aab5477da36dae4973597136",
            "IsHidden": "False",
        },
        {
            "Start": "1429701dc32f4e24bf066f6f4d2124c9",
            "End": "ba59a996b5d44c6292d22a4c0d268cf6",
            "Id": "45b4fcf138fd4e0a8ae7380986d8e725",
            "IsHidden": "False",
        },
        {
            "Start": "7b0fe2bd502a4b1a916cb2ceb66e2ea3",
            "End": "6a1dc87305dc45c5a5647172ffab2d09",
            "Id": "8cd4d1887832460e8189e5419441bb95",
            "IsHidden": "False",
        },
        {
            "Start": "49f51eb983444badb462bd6b8a4068b8",
            "End": "55791089ba5548749646958a7842f63e",
            "Id": "7c8c41313f3448a6a0ea1c828a4d87ee",
            "IsHidden": "False",
        },
        {
            "Start": "6065416333634a059b54285277119ca5",
            "End": "8d6daf8e6d9547d6a41a8e9c93962a30",
            "Id": "f174d8cf086c4e4996536b6d22770d72",
            "IsHidden": "False",
        },
        {
            "Start": "7ea795726aa14f5da79c043428d299f4",
            "End": "6199c4780c3044ef812db759d8e9f428",
            "Id": "a74f23dab96745d887910d0e09e5fd82",
            "IsHidden": "False",
        },
        {
            "Start": "cfb7a7b17c124f1d8291a415f5a3b1a7",
            "End": "5994a827c1924f2da2fe54e7e048efe2",
            "Id": "201e24d8eb774e0a961f67f409f49455",
            "IsHidden": "False",
        },
    ]

    Group_template = {
        "Id": "ff8a44f7ea6d4bc3bbd6dcf6fb68f081",
        "Title": "NumberOfInstances",
        "DescriptionText": "说明 <双击此处编辑组说明>",
        "IsExpanded": True,
        "WidthAdjustment": 124.76651670452861,
        "HeightAdjustment": 2.4512715762812149,
        "Nodes": [],
        "HasNestedGroups": False,
        "Left": -1003.7002359109599,
        "Top": -371.84226681343171,
        "Width": 1292.2646673173115,
        "Height": 913.14804946951472,
        "FontSize": 36.0,
        "GroupStyleId": "00000000-0000-0000-0000-000000000000",
        "InitialTop": -299.44226681343173,
        "InitialHeight": 595.754234721797,
        "TextblockHeight": 62.4,
        "Background": "#FFC1D676",
    }

    Node_list_template2 = copy.deepcopy(Node_list_template)
    Node_list_template2[2] = {
        "ConcreteType": "Dynamo.Graph.Nodes.ZeroTouch.DSFunction, DynamoCore",
        "Id": "a5e851ffc1694a80a96721988458f021",
        "NodeType": "FunctionNode",
        "Inputs": [
            {
                "Id": "c080a813e711475da79bd28e50fd84f6",
                "Name": "familyType",
                "Description": "族类型。也称为族符号。\n\nFamilyType",
                "UsingDefaultValue": False,
                "Level": 2,
                "UseLevels": False,
                "KeepListStructure": False,
            },
            {
                "Id": "6199c4780c3044ef812db759d8e9HOST",
                "Name": "host",
                "Description": "将向其插入 FamilyInstance 的对象通常称为主体。\n\nElement",
                "UsingDefaultValue": False,
                "Level": 2,
                "UseLevels": False,
                "KeepListStructure": False,
            },
            {
                "Id": "55791089ba5548749646958a7842f63e",
                "Name": "point",
                "Description": "将放置实例的物理位置。\n\nPoint",
                "UsingDefaultValue": False,
                "Level": 2,
                "UseLevels": False,
                "KeepListStructure": False,
            },
        ],
        "Outputs": [
            {
                "Id": "6065416333634a059b54285277119ca5",
                "Name": "FamilyInstance",
                "Description": "FamilyInstance",
                "UsingDefaultValue": False,
                "Level": 2,
                "UseLevels": False,
                "KeepListStructure": False,
            }
        ],
        "FunctionSignature": "Revit.Elements.FamilyInstance.ByHostAndPoint@Revit.Elements.FamilyType,Revit.Elements.Element,Autodesk.DesignScript.Geometry.Point",
        "Replication": "Auto",
        "Description": "根据 FamilyType (在 Revit API 中也称为 FamilySymbol)及其主体图元和位置放置 Revit FamilyInstance。\n\nFamilyInstance.ByHostAndPoint (familyType: FamilyType, host: Element, point: Point): FamilyInstance",
    }

    MAPPING = {}
    NODES = []
    CONNECTOR = []

    transformation_list_by_type = {}
    for id in transformation_dict:
        family_type = id2mesh[id][1].type
        if family_type not in transformation_list_by_type:
            transformation_list_by_type[family_type] = []
        tmp1 = transformation_dict[id]
        tmp2 = []
        tmp3 = []
        for param in tmp1:
            mesh = id2mesh[id][0].mesh
            mesh_center = mesh.get_center()
            translation = (
                param["c"][0] - mesh_center[0],
                param["c"][1] - mesh_center[1],
                param["c"][2] - mesh_center[2],
            )
            rotation = param["best_rz"]
            x, y, z = translation
            rz = -np.rad2deg(rotation)

            if id in host_relationship:
                tmp3.append([x, y, z, rz, id])
            else:
                tmp2.append([x, y, z, rz, id])
        if len(tmp2) > 0:
            transformation_list_by_type[family_type].extend(tmp2)
        if len(tmp3) > 0:
            if family_type + "_with_host" not in transformation_list_by_type:
                transformation_list_by_type[family_type + "_with_host"] = []
            transformation_list_by_type[family_type + "_with_host"].extend(tmp3)

    transformation_list_by_type = {
        i: transformation_list_by_type[i]
        for i in transformation_list_by_type
        if len(transformation_list_by_type[i]) > 0
    }

    host_store = [{}, {}]  # 0: output, 1: input
    for comp in transformation_list_by_type:
        transformation_list = transformation_list_by_type[comp]

        Node_family_type = copy.deepcopy(Node_family_template)
        Node_family_type["Id"] = str(uuid.uuid4().hex)
        for input in Node_family_type["Inputs"]:
            new_id = str(uuid.uuid4().hex)
            MAPPING[input["Id"]] = new_id
            input["Id"] = new_id
        for output in Node_family_type["Outputs"]:
            new_id = str(uuid.uuid4().hex)
            MAPPING[output["Id"]] = new_id
            output["Id"] = new_id
        NODES.append(Node_family_type)

        if not "with_host" in comp.lower():
            for node in Node_level_template:
                node = copy.deepcopy(node)
                node["Id"] = str(uuid.uuid4().hex)
                for input in node["Inputs"]:
                    new_id = str(uuid.uuid4().hex)
                    MAPPING[input["Id"]] = new_id
                    input["Id"] = new_id
                for output in node["Outputs"]:
                    new_id = str(uuid.uuid4().hex)
                    MAPPING[output["Id"]] = new_id
                    output["Id"] = new_id
                NODES.append(node)

        for i in range(len(transformation_list)):
            x, y, z, rz, mesh_id = transformation_list[i]

            node_group = []
            for node in (
                Node_list_template
                if not "with_host" in comp.lower()
                else Node_list_template2
            ):
                node = copy.deepcopy(node)
                node["Id"] = str(uuid.uuid4().hex)
                for input in node["Inputs"]:
                    new_id = str(uuid.uuid4().hex)
                    MAPPING[input["Id"]] = new_id
                    input["Id"] = new_id
                    if "with_host" in comp.lower() and input["Name"] == "host":
                        host_store[1][mesh_id] = input["Id"]
                for output in node["Outputs"]:
                    new_id = str(uuid.uuid4().hex)
                    MAPPING[output["Id"]] = new_id
                    output["Id"] = new_id
                    if (
                        comp.lower() == "column"
                        and output["Name"] == "FamilyInstance"
                    ):
                        host_store[0][mesh_id] = (output["Id"], x, y, z)
                if node["NodeType"] == "CodeBlockNode":
                    node["Code"] = f"{x * 1000};\n{y * 1000};\n0;\n{rz};"
                    if "with_host" in comp.lower():
                        _, _x, _y, _z = host_store[0][
                            host_relationship[mesh_id]
                        ]

                        column_o3d_mesh = id2mesh[host_relationship[mesh_id]][
                            0
                        ].mesh
                        height = (
                            column_o3d_mesh.get_max_bound()[2]
                            - column_o3d_mesh.get_min_bound()[2]
                        )
                        print(
                            column_o3d_mesh.get_max_bound()[2],
                            column_o3d_mesh.get_min_bound()[2],
                            height,
                        )
                        node["Code"] = (
                            f"{_x * 1000};\n{_y * 1000};\n{height * 1000};\n{rz};"
                        )

                node_group.append(node)
            NODES.extend(node_group)
            for connector in Connector_list_template:
                connector = copy.deepcopy(connector)
                if (
                    connector["Start"] not in MAPPING
                    or connector["End"] not in MAPPING
                ):
                    continue
                connector["Start"] = MAPPING[connector["Start"]]
                connector["End"] = MAPPING[connector["End"]]
                connector["Id"] = str(uuid.uuid4().hex)
                CONNECTOR.append(connector)

            group = Group_template.copy()
            group["Id"] = str(uuid.uuid4().hex)
            group["Title"] = f"{comp} #{i}"
            group["Nodes"] = [node["Id"] for node in node_group]
            root["View"]["Annotations"].append(group)
    for mesh_id in host_relationship:
        CONNECTOR.append(
            {
                "Start": host_store[0][host_relationship[mesh_id]][0],
                "End": host_store[1][mesh_id],
                "Id": str(uuid.uuid4().hex),
                "IsHidden": "False",
            }
        )
    root["Nodes"] = NODES
    root["Connectors"] = CONNECTOR
    return root


def export_dynamo(history, id2mesh, host_relationship):
    return family_instance_generate_node(history, id2mesh, host_relationship)


def main():
    transformation_list = [[10, 0, 0, 0], [0, 10, 0, 0]]
    data = json.dumps(
        family_instance_generate_node(transformation_list), indent=4
    )
    with open("./family_instance_generate.dyn", "w") as f:
        f.write(data)


if __name__ == "__main__":
    main()
