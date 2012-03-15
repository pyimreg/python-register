// import QtQuick 1.0 // to target S60 5th Edition or Maemo 5
import QtQuick 1.1

Rectangle {
    id: rectangle1
    width: 1000
    height: 500
    MouseArea {
        id: main_mouse_area
        anchors.fill: parent
        onClicked: {
            Qt.quit();
        }

        Image {
            id: left_image
            anchors.rightMargin: main_mouse_area.width/2 + 25
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
            anchors.leftMargin: main_mouse_area.width/2 + 25
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

    }


}
