import QtQuick 1.1
import Qt 4.7

Rectangle {    
    id: main_rectangle
    width: 1000
    height: 500

    signal open_left_clicked;
    signal open_right_clicked;

    function open_left(url) {
        left_image.source = url
    }

    function open_right(url) {
        left_image.source = url
    }

    Image {
        id: left_image
        anchors.rightMargin: parent.width/2 + 25
        anchors.right: parent.right
        anchors.left: parent.left
        anchors.leftMargin: 50
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 50
        anchors.top: parent.top
        anchors.topMargin: 50
        source: "africa-map1.jpg"
    }

    Image {
        id: right_image
        anchors.right: parent.right
        anchors.rightMargin: 50
        anchors.leftMargin: parent.width/2 + 25
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 50
        anchors.top: parent.top
        anchors.topMargin: 50
        source: "africa-map1.jpg"
    }

    Image {
        id: left_image_zoom
        width: left_image.width / 5
        height: left_image.height / 5
        anchors.right: left_image.right
        anchors.rightMargin: 5
        anchors.top: left_image.top
        anchors.topMargin: 5
        source: "africa-map1.jpg"
    }

    Image {
        id: right_image_zoom
        width: right_image.width / 5
        height: right_image.height / 5
        anchors.left: right_image.left
        anchors.leftMargin: 5
        anchors.top: right_image.top
        anchors.topMargin: 5
        source: "africa-map1.jpg"
    }


    Flow {
        id: menu_flow
        height: 45
        spacing: 0
        anchors.right: parent.right
        anchors.rightMargin: 0
        anchors.left: parent.left
        anchors.leftMargin: 0
        anchors.top: parent.top
        anchors.topMargin: 0

        Text {
            id: menu_open_left
            x: 26
            y: 30
            width: 100
            height: 45
            text: qsTr("Open Left...")
            verticalAlignment: Text.AlignVCenter
            font.pixelSize: 12
            MouseArea {
                id: menu_open_left_mouse_area
                anchors.fill: parent
                onClicked: open_left_clicked()
            }
        }

        Text {
            id: menu_open_right
            x: 26
            y: 30
            width: 100
            height: 45
            text: qsTr("Open Right...")
            verticalAlignment: Text.AlignVCenter
            font.pixelSize: 12
            MouseArea {
                id: menu_open_right_mouse_area
                anchors.fill: parent
                onClicked: open_right_clicked()
            }
        }
    }

}
